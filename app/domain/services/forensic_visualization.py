"""Motor de Visualizacao Forense para Analise de Audio Deepfake.

Modulo puro de visualizacao (sem dependencia do Gradio).
Todas as funcoes retornam matplotlib.figure.Figure usando a API OOP (thread-safe).
"""

import logging
from typing import Any, Dict, List, Tuple

import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)


# =============================================================================
# Forensic Theme
# =============================================================================

class ForensicTheme:
    """Tema visual escuro consistente para todas as visualizacoes forenses.

    Paleta alinhada com o design system XfakeSong (slate/blue dark theme).
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
        ax1.set_ylabel('Frequencia (Hz)')
        fig.colorbar(img1, ax=ax1, format='%+.0f dB', fraction=0.046)

        # STFT Magnitude
        ax2 = fig.add_subplot(2, 2, 2)
        D = librosa.stft(y, n_fft=2048, hop_length=512)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        img2 = ax2.imshow(S_db, aspect='auto', origin='lower', cmap=cmap,
                          extent=[0, len(y) / self.sr, 0, self.sr / 2])
        ax2.set_title('STFT Magnitude (dB)')
        ax2.set_ylabel('Frequencia (Hz)')
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
            S_linear = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
            S_power = S_linear ** 2
            # Linear filterbank (uniform spacing)
            n_filters = 64
            n_fft_bins = S_power.shape[0]
            filterbank = np.zeros((n_filters, n_fft_bins))
            bin_width = n_fft_bins // n_filters
            for i in range(n_filters):
                start = i * bin_width
                end = min(start + bin_width, n_fft_bins)
                filterbank[i, start:end] = 1.0
            lfcc_spec = filterbank @ S_power
            lfcc_spec = librosa.power_to_db(lfcc_spec + 1e-10, ref=np.max)
            from scipy.fft import dct
            lfcc = dct(lfcc_spec, type=2, axis=0, norm='ortho')[:20, :]
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

        fig.suptitle('Comparacao Multi-Espectrograma', fontsize=14,
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
        ax.set_ylabel('Frequencia (Hz)')
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

        # Extract formants via LPC
        try:
            frame_len = int(0.025 * self.sr)  # 25ms frames
            hop = int(0.010 * self.sr)  # 10ms hop
            lpc_order = 2 + int(self.sr / 1000)

            formant_tracks = {f'F{i+1}': [] for i in range(4)}
            times = []

            for start in range(0, len(y) - frame_len, hop):
                frame = y[start:start + frame_len]
                frame = frame * np.hamming(len(frame))

                # LPC via autocorrelation
                try:
                    a = librosa.lpc(frame, order=lpc_order)
                    roots = np.roots(a)
                    roots = roots[np.imag(roots) >= 0]
                    angles = np.angle(roots)
                    freqs = sorted(angles * (self.sr / (2 * np.pi)))
                    freqs = [f for f in freqs if 90 < f < self.sr / 2]

                    for i in range(4):
                        if i < len(freqs):
                            formant_tracks[f'F{i+1}'].append(freqs[i])
                        else:
                            formant_tracks[f'F{i+1}'].append(np.nan)
                    times.append((start + frame_len / 2) / self.sr)
                except Exception:
                    for i in range(4):
                        formant_tracks[f'F{i+1}'].append(np.nan)
                    times.append((start + frame_len / 2) / self.sr)

            # Plot formant tracks
            colors = [ForensicTheme.ACCENT_CYAN, ForensicTheme.ACCENT_ORANGE,
                      ForensicTheme.ACCENT_GREEN, ForensicTheme.ACCENT_RED]
            for i, (name, track) in enumerate(formant_tracks.items()):
                ax.plot(times, track, color=colors[i], linewidth=1.5,
                        alpha=0.8, label=name)

            ax.legend(loc='upper right', fontsize=9,
                      facecolor=ForensicTheme.BG_COLOR, labelcolor=ForensicTheme.TEXT_COLOR)
        except Exception as e:
            logger.warning(f"Formant extraction failed: {e}")

        ax.set_title('Envelope Espectral & Formantes')
        ax.set_xlabel('Tempo (s)')
        ax.set_ylabel('Frequencia (Hz)')

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

        frame_len = int(0.025 * self.sr)
        hop = int(0.010 * self.sr)

        hnr_values = []
        times = []

        for start in range(0, len(y) - frame_len, hop):
            frame = y[start:start + frame_len]
            # Autocorrelation-based HNR
            try:
                autocorr = np.correlate(frame, frame, mode='full')
                autocorr = autocorr[len(autocorr) // 2:]
                if autocorr[0] > 0:
                    autocorr = autocorr / autocorr[0]
                    # Find first peak after lag 0
                    min_lag = int(self.sr / 500)  # 500 Hz max
                    max_lag = int(self.sr / 50)   # 50 Hz min
                    max_lag = min(max_lag, len(autocorr) - 1)
                    if min_lag < max_lag:
                        peak = np.max(autocorr[min_lag:max_lag])
                        if peak > 0 and peak < 1:
                            hnr = 10 * np.log10(peak / (1 - peak + 1e-10))
                            hnr_values.append(np.clip(hnr, -10, 40))
                        else:
                            hnr_values.append(0.0)
                    else:
                        hnr_values.append(0.0)
                else:
                    hnr_values.append(0.0)
            except Exception:
                hnr_values.append(0.0)
            times.append((start + frame_len / 2) / self.sr)

        hnr_values = np.array(hnr_values)
        times = np.array(times)

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
        import librosa

        fig = Figure(figsize=(14, 5))

        # Extract F0 periods for jitter/shimmer computation
        try:
            f0, voiced_flag, voiced_probs = librosa.pyin(
                y, fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7'),
                sr=self.sr)
            times = librosa.times_like(f0, sr=self.sr)

            # Voiced frames only
            voiced_idx = ~np.isnan(f0)
            f0_voiced = f0[voiced_idx]
            times_voiced = times[voiced_idx]

            if len(f0_voiced) < 3:
                ax = fig.add_subplot(111)
                ax.text(0.5, 0.5, 'Insuficientes frames vocais para analise',
                        ha='center', va='center', color=ForensicTheme.TEXT_COLOR,
                        transform=ax.transAxes)
                ForensicTheme.apply(fig)
                return fig

            # Compute local jitter (period-to-period variation)
            periods = 1.0 / (f0_voiced + 1e-10)
            jitter_local = np.abs(np.diff(periods)) / (np.mean(periods) + 1e-10) * 100

            # Compute local shimmer (amplitude variation between consecutive voiced frames)
            amplitudes = []
            hop_length = 512
            for idx in np.where(voiced_idx)[0]:
                center = idx * hop_length
                start = max(0, center - hop_length // 2)
                end = min(len(y), center + hop_length // 2)
                amplitudes.append(np.max(np.abs(y[start:end])))
            amplitudes = np.array(amplitudes)
            shimmer_local = np.abs(np.diff(amplitudes)) / (np.mean(amplitudes) + 1e-10) * 100

            # Left: Jitter
            ax1 = fig.add_subplot(1, 2, 1)
            ax1.plot(times_voiced[1:], jitter_local,
                     color=ForensicTheme.ACCENT_CYAN, linewidth=0.8, alpha=0.7)
            jitter_mean = np.mean(jitter_local)
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
            ax2.plot(times_voiced[1:len(shimmer_local) + 1], shimmer_local,
                     color=ForensicTheme.ACCENT_GREEN, linewidth=0.8, alpha=0.7)
            shimmer_mean = np.mean(shimmer_local)
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

        fig.suptitle('Analise de Micro-Perturbacoes Vocais',
                     color=ForensicTheme.TEXT_COLOR, fontweight='bold')
        ForensicTheme.apply(fig)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        return fig

    # -----------------------------------------------------------------
    # 10. F0 Detailed Analysis
    # -----------------------------------------------------------------
    def plot_f0_detailed(self, y: np.ndarray) -> Figure:
        """Pitch contour (pyin) com banda de confianca e vibrato."""
        import librosa

        fig = Figure(figsize=(14, 6))

        try:
            f0, voiced_flag, voiced_probs = librosa.pyin(
                y, fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7'),
                sr=self.sr)
            times = librosa.times_like(f0, sr=self.sr)

            # Main pitch contour
            ax1 = fig.add_subplot(2, 1, 1)
            voiced_mask = ~np.isnan(f0)

            # Plot with confidence coloring
            ax1.scatter(times[voiced_mask], f0[voiced_mask],
                        c=voiced_probs[voiced_mask], cmap='RdYlGn',
                        s=3, alpha=0.8, vmin=0, vmax=1)
            ax1.plot(times, f0, color=ForensicTheme.ACCENT_CYAN,
                     linewidth=0.8, alpha=0.5)

            # Statistics
            f0_voiced = f0[voiced_mask]
            if len(f0_voiced) > 0:
                f0_mean = np.mean(f0_voiced)
                f0_std = np.std(f0_voiced)
                ax1.axhline(y=f0_mean, color=ForensicTheme.ACCENT_ORANGE,
                            linestyle='--', alpha=0.7,
                            label=f'F0 medio: {f0_mean:.1f} Hz')
                ax1.fill_between(times, f0_mean - f0_std, f0_mean + f0_std,
                                 alpha=0.1, color=ForensicTheme.ACCENT_ORANGE)

            ax1.set_title('Contorno de Pitch (F0) com Confianca')
            ax1.set_ylabel('Frequencia (Hz)')
            ax1.legend(facecolor=ForensicTheme.BG_COLOR,
                       labelcolor=ForensicTheme.TEXT_COLOR, fontsize=9)

            # Vibrato analysis (bottom)
            ax2 = fig.add_subplot(2, 1, 2)

            if len(f0_voiced) > 10:
                # Compute vibrato rate via short-term F0 fluctuation
                f0_interp = np.interp(times, times[voiced_mask], f0_voiced)
                f0_detrended = f0_interp - np.convolve(
                    f0_interp, np.ones(20) / 20, mode='same')

                ax2.plot(times, f0_detrended, color=ForensicTheme.ACCENT_GREEN,
                         linewidth=0.8, alpha=0.8)
                ax2.fill_between(times, f0_detrended, alpha=0.2,
                                 color=ForensicTheme.ACCENT_GREEN)

                # Vibrato extent
                vibrato_extent = np.std(f0_detrended)
                ax2.axhline(y=vibrato_extent, color=ForensicTheme.ACCENT_ORANGE,
                            linestyle='--', alpha=0.5,
                            label=f'Extent: +/-{vibrato_extent:.1f} Hz')
                ax2.axhline(y=-vibrato_extent, color=ForensicTheme.ACCENT_ORANGE,
                            linestyle='--', alpha=0.5)

            ax2.set_title('Analise de Vibrato (Flutuacao F0)')
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

        # Normalize
        importances = np.array(importances)
        if np.max(importances) > 0:
            importances = importances / np.max(importances)

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
        ax.set_title('Importancia Relativa de Features',
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
        ax.set_ylabel('Frequencia (Hz)')

        ForensicTheme.apply(fig)
        fig.tight_layout()
        return fig

    # -----------------------------------------------------------------
    # 13. Confidence Timeline
    # -----------------------------------------------------------------
    def plot_confidence_timeline(self, segment_times: np.ndarray,
                                 segment_confidences: np.ndarray) -> Figure:
        """Linha de confianca por segmento com threshold."""
        fig = Figure(figsize=(12, 4))
        ax = fig.add_subplot(111)

        ax.plot(segment_times, segment_confidences,
                color=ForensicTheme.ACCENT_CYAN, linewidth=2, marker='o',
                markersize=4)
        ax.fill_between(segment_times, segment_confidences,
                        alpha=0.2, color=ForensicTheme.ACCENT_CYAN)

        # Threshold line
        ax.axhline(y=0.5, color=ForensicTheme.ACCENT_RED, linestyle='--',
                   linewidth=1.5, alpha=0.8, label='Threshold (0.5)')

        # Color regions
        above = segment_confidences > 0.5
        ax.fill_between(segment_times, 0.5, segment_confidences,
                        where=above, alpha=0.15, color=ForensicTheme.ACCENT_RED,
                        label='Fake (>0.5)')
        ax.fill_between(segment_times, segment_confidences, 0.5,
                        where=~above, alpha=0.15, color=ForensicTheme.ACCENT_GREEN,
                        label='Real (<0.5)')

        ax.set_ylim(-0.05, 1.05)
        ax.set_title('Confianca de Deteccao por Segmento')
        ax.set_xlabel('Tempo (s)')
        ax.set_ylabel('Confianca (Fake)')
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
    # 2. Gradient Norm Tracking
    # -----------------------------------------------------------------
    def plot_gradient_norms(self, grad_norms: List[float]) -> Figure:
        """Norma L2 dos gradientes por epoca."""
        fig = Figure(figsize=(10, 4))
        ax = fig.add_subplot(111)

        epochs = list(range(1, len(grad_norms) + 1))
        ax.plot(epochs, grad_norms, color=ForensicTheme.ACCENT_ORANGE,
                linewidth=1.5)
        ax.fill_between(epochs, grad_norms, alpha=0.2,
                        color=ForensicTheme.ACCENT_ORANGE)

        # Warning threshold
        mean_norm = np.mean(grad_norms)
        ax.axhline(y=mean_norm * 3, color=ForensicTheme.ACCENT_RED,
                   linestyle='--', alpha=0.5, label=f'3x media ({mean_norm * 3:.2f})')

        ax.set_title('Norma dos Gradientes por Epoca')
        ax.set_xlabel('Epoca')
        ax.set_ylabel('Norma L2')
        ax.legend(facecolor=ForensicTheme.BG_COLOR,
                  labelcolor=ForensicTheme.TEXT_COLOR)

        ForensicTheme.apply(fig)
        fig.tight_layout()
        return fig

    # -----------------------------------------------------------------
    # 3. Per-Class Accuracy
    # -----------------------------------------------------------------
    def plot_per_class_accuracy(self, real_acc: List[float],
                                 fake_acc: List[float]) -> Figure:
        """Acuracia separada real/fake por epoca."""
        fig = Figure(figsize=(10, 5))
        ax = fig.add_subplot(111)

        epochs = list(range(1, len(real_acc) + 1))
        ax.plot(epochs, real_acc, color=ForensicTheme.ACCENT_GREEN,
                linewidth=2, label='Real', marker='o', markersize=3)
        ax.plot(epochs, fake_acc, color=ForensicTheme.ACCENT_RED,
                linewidth=2, label='Fake', marker='s', markersize=3)

        # Overall average
        overall = [(r + f) / 2 for r, f in zip(real_acc, fake_acc)]
        ax.plot(epochs, overall, color=ForensicTheme.ACCENT_CYAN,
                linewidth=1.5, linestyle='--', label='Media', alpha=0.7)

        ax.set_ylim(0, 1.05)
        ax.set_title('Acuracia por Classe ao Longo do Treinamento')
        ax.set_xlabel('Epoca')
        ax.set_ylabel('Acuracia')
        ax.legend(facecolor=ForensicTheme.BG_COLOR,
                  labelcolor=ForensicTheme.TEXT_COLOR)

        ForensicTheme.apply(fig)
        fig.tight_layout()
        return fig

    # -----------------------------------------------------------------
    # 4. DET Curve + EER
    # -----------------------------------------------------------------
    def plot_det_curve_eer(self, y_true: np.ndarray,
                           y_scores: np.ndarray) -> Tuple[Figure, float]:
        """Curva DET (FNR vs FPR) com ponto EER. Retorna (Figure, eer_value)."""
        from sklearn.metrics import roc_curve

        fig = Figure(figsize=(8, 8))
        ax = fig.add_subplot(111)

        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        fnr = 1 - tpr

        # Find EER (where FPR ~= FNR)
        try:
            from scipy.interpolate import interp1d
            from scipy.optimize import brentq
            eer_fn = interp1d(fpr, fnr)
            eer = brentq(lambda x: eer_fn(x) - x, 0.001, 0.999)
        except Exception:
            # Fallback: find closest point
            diff = np.abs(fpr - fnr)
            eer_idx = np.argmin(diff)
            eer = (fpr[eer_idx] + fnr[eer_idx]) / 2

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
        return fig, float(eer)

    # -----------------------------------------------------------------
    # 5. Threshold Optimization
    # -----------------------------------------------------------------
    def plot_threshold_optimization(self, y_true: np.ndarray,
                                     y_scores: np.ndarray) -> Figure:
        """Precision, Recall, F1 plotados vs threshold."""
        from sklearn.metrics import f1_score, precision_score, recall_score

        fig = Figure(figsize=(10, 5))
        ax = fig.add_subplot(111)

        thresholds = np.arange(0.05, 0.96, 0.01)
        precisions, recalls, f1s = [], [], []

        for t in thresholds:
            y_pred = (y_scores >= t).astype(int)
            precisions.append(precision_score(y_true, y_pred, zero_division=0))
            recalls.append(recall_score(y_true, y_pred, zero_division=0))
            f1s.append(f1_score(y_true, y_pred, zero_division=0))

        ax.plot(thresholds, precisions, color=ForensicTheme.ACCENT_CYAN,
                linewidth=2, label='Precision')
        ax.plot(thresholds, recalls, color=ForensicTheme.ACCENT_ORANGE,
                linewidth=2, label='Recall')
        ax.plot(thresholds, f1s, color=ForensicTheme.ACCENT_GREEN,
                linewidth=2.5, label='F1-Score')

        # Best F1 threshold
        best_idx = np.argmax(f1s)
        best_threshold = thresholds[best_idx]
        ax.axvline(x=best_threshold, color=ForensicTheme.ACCENT_RED,
                   linestyle='--', alpha=0.7,
                   label=f'Melhor F1 @ {best_threshold:.2f}')

        ax.set_title('Otimizacao de Threshold de Decisao')
        ax.set_xlabel('Threshold')
        ax.set_ylabel('Score')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.legend(facecolor=ForensicTheme.BG_COLOR,
                  labelcolor=ForensicTheme.TEXT_COLOR)

        ForensicTheme.apply(fig)
        fig.tight_layout()
        return fig

    # -----------------------------------------------------------------
    # 6. Distribution Comparison
    # -----------------------------------------------------------------
    def plot_distribution_comparison(self, train_features: np.ndarray,
                                      val_features: np.ndarray,
                                      feature_name: str = "Feature") -> Figure:
        """Histogramas sobrepostos train vs val."""
        fig = Figure(figsize=(10, 5))
        ax = fig.add_subplot(111)

        train_flat = train_features.flatten()
        val_flat = val_features.flatten()

        bins = np.linspace(
            min(train_flat.min(), val_flat.min()),
            max(train_flat.max(), val_flat.max()),
            60
        )

        ax.hist(train_flat, bins=bins, alpha=0.5,
                color=ForensicTheme.ACCENT_CYAN, label='Treino', density=True)
        ax.hist(val_flat, bins=bins, alpha=0.5,
                color=ForensicTheme.ACCENT_ORANGE, label='Validacao', density=True)

        ax.set_title(f'Distribuicao de {feature_name}: Treino vs Validacao')
        ax.set_xlabel('Valor')
        ax.set_ylabel('Densidade')
        ax.legend(facecolor=ForensicTheme.BG_COLOR,
                  labelcolor=ForensicTheme.TEXT_COLOR)

        ForensicTheme.apply(fig)
        fig.tight_layout()
        return fig

    # -----------------------------------------------------------------
    # 7. Model Comparison
    # -----------------------------------------------------------------
    def plot_model_comparison(self,
                              models_metrics: Dict[str, Dict[str, float]]) -> Figure:
        """Barras agrupadas comparando multiplos modelos."""
        fig = Figure(figsize=(12, 6))
        ax = fig.add_subplot(111)

        model_names = list(models_metrics.keys())
        if not model_names:
            ax.text(0.5, 0.5, 'Sem dados de comparacao',
                    ha='center', va='center', color=ForensicTheme.TEXT_COLOR,
                    transform=ax.transAxes)
            ForensicTheme.apply(fig)
            return fig

        # Common metrics
        metric_keys = ['accuracy', 'precision', 'recall', 'f1_score']
        metric_labels = ['Acuracia', 'Precisao', 'Recall', 'F1']

        x = np.arange(len(metric_labels))
        width = 0.8 / len(model_names)

        for i, model_name in enumerate(model_names):
            values = [models_metrics[model_name].get(k, 0) for k in metric_keys]
            color = ForensicTheme.PALETTE[i % len(ForensicTheme.PALETTE)]
            ax.bar(x + i * width, values, width, label=model_name,
                   color=color, alpha=0.85)

        ax.set_xticks(x + width * (len(model_names) - 1) / 2)
        ax.set_xticklabels(metric_labels)
        ax.set_ylim(0, 1.1)
        ax.set_title('Comparacao de Modelos')
        ax.set_ylabel('Score')
        ax.legend(facecolor=ForensicTheme.BG_COLOR,
                  labelcolor=ForensicTheme.TEXT_COLOR, fontsize=9)

        ForensicTheme.apply(fig)
        fig.tight_layout()
        return fig

    # -----------------------------------------------------------------
    # 8. t-SNE / UMAP Embedding
    # -----------------------------------------------------------------
    def plot_embedding_2d(self, features: np.ndarray,
                          labels: np.ndarray,
                          method: str = "tsne") -> Figure:
        """Scatter 2D de representacoes aprendidas coloridas por classe."""
        fig = Figure(figsize=(8, 8))
        ax = fig.add_subplot(111)

        # Subsample if too large
        max_samples = 3000
        if len(features) > max_samples:
            idx = np.random.choice(len(features), max_samples, replace=False)
            features = features[idx]
            labels = labels[idx]

        # Flatten features if needed
        if features.ndim > 2:
            features = features.reshape(features.shape[0], -1)

        try:
            if method == "umap":
                try:
                    import umap
                    reducer = umap.UMAP(n_components=2, random_state=42)
                    embedding = reducer.fit_transform(features)
                except ImportError:
                    method = "tsne"

            if method == "tsne":
                from sklearn.manifold import TSNE
                perplexity = min(30, len(features) - 1)
                tsne = TSNE(n_components=2, random_state=42,
                            perplexity=max(5, perplexity))
                embedding = tsne.fit_transform(features)

            # Plot
            unique_labels = np.unique(labels)
            label_names = {0: 'Real', 1: 'Fake'}
            colors = {0: ForensicTheme.ACCENT_GREEN, 1: ForensicTheme.ACCENT_RED}

            for label in unique_labels:
                mask = labels == label
                name = label_names.get(int(label), f'Classe {label}')
                color = colors.get(int(label), ForensicTheme.ACCENT_CYAN)
                ax.scatter(embedding[mask, 0], embedding[mask, 1],
                           c=color, label=name, alpha=0.6, s=15, edgecolors='none')

            ax.set_title(f'Embedding 2D ({method.upper()})')
            ax.set_xlabel('Componente 1')
            ax.set_ylabel('Componente 2')
            ax.legend(facecolor=ForensicTheme.BG_COLOR,
                      labelcolor=ForensicTheme.TEXT_COLOR)

        except Exception as e:
            ax.text(0.5, 0.5, f'Erro no embedding:\n{e}',
                    ha='center', va='center', color=ForensicTheme.TEXT_COLOR,
                    transform=ax.transAxes)

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

        ax.set_title(f'Distribuicao de Resultados (Total: {total})',
                     color=ForensicTheme.TEXT_COLOR, fontweight='bold')
        fig.patch.set_facecolor(ForensicTheme.BG_COLOR)
        fig.tight_layout()
        return fig

    # -----------------------------------------------------------------
    # 2. Confidence Distribution
    # -----------------------------------------------------------------
    def plot_confidence_distribution(self, confidences: np.ndarray) -> Figure:
        """Histograma de scores de confianca."""
        fig = Figure(figsize=(10, 5))
        ax = fig.add_subplot(111)

        if len(confidences) == 0:
            ax.text(0.5, 0.5, 'Sem dados',
                    ha='center', va='center', color=ForensicTheme.TEXT_COLOR,
                    transform=ax.transAxes)
            ForensicTheme.apply(fig)
            return fig

        bins = np.linspace(0, 1, 30)
        n, bins_out, patches = ax.hist(confidences, bins=bins, edgecolor='black',
                                        alpha=0.8, linewidth=0.5)

        # Color bars by value
        anomaly_cmap = ForensicTheme.get_anomaly_cmap()
        for patch, left_edge in zip(patches, bins_out[:-1]):
            patch.set_facecolor(anomaly_cmap(left_edge))

        # Threshold line
        ax.axvline(x=0.5, color='white', linestyle='--', linewidth=2,
                   label='Threshold (0.5)')

        # Stats
        mean_conf = np.mean(confidences)
        ax.axvline(x=mean_conf, color=ForensicTheme.ACCENT_CYAN,
                   linestyle=':', linewidth=1.5,
                   label=f'Media: {mean_conf:.3f}')

        ax.set_title('Distribuicao de Scores de Confianca')
        ax.set_xlabel('Confianca (Fake)')
        ax.set_ylabel('Contagem')
        ax.legend(facecolor=ForensicTheme.BG_COLOR,
                  labelcolor=ForensicTheme.TEXT_COLOR)

        ForensicTheme.apply(fig)
        fig.tight_layout()
        return fig

    # -----------------------------------------------------------------
    # 3. Generate Report Data
    # -----------------------------------------------------------------
    def generate_forensic_report_data(self,
                                       results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Dados agregados para exportacao de relatorio."""
        if not results:
            return {"total": 0, "summary": "Sem resultados"}

        n_fake = sum(1 for r in results if r.get('is_fake', False))
        n_real = len(results) - n_fake
        confidences = [r.get('confidence', 0) for r in results]

        return {
            "total_analyzed": len(results),
            "total_fake": n_fake,
            "total_real": n_real,
            "fake_percentage": n_fake / len(results) * 100,
            "confidence_mean": float(np.mean(confidences)),
            "confidence_std": float(np.std(confidences)),
            "confidence_min": float(np.min(confidences)),
            "confidence_max": float(np.max(confidences)),
            "high_confidence_fakes": sum(
                1 for r in results if r.get('is_fake') and r.get('confidence', 0) > 0.8
            ),
            "low_confidence_detections": sum(
                1 for r in results if 0.4 < r.get('confidence', 0) < 0.6
            ),
            "per_file": [
                {
                    "filename": r.get('filename', 'unknown'),
                    "is_fake": r.get('is_fake', False),
                    "confidence": r.get('confidence', 0),
                    "model": r.get('model_name', 'unknown')
                }
                for r in results
            ]
        }
