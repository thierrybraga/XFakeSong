"""Medidas forenses de áudio: o que os gráficos da aba Investigar exibem.

Só cálculo (numpy, librosa, scipy, sklearn), sem matplotlib: a plotagem vive
em ``app/interfaces/gradio/presenters/forensic_plots.py`` (P1-7). Aqui fica
toda medida com algoritmo próprio — formantes por LPC, HNR, jitter/shimmer,
F0 e vibrato, LFCC, DET/EER, varredura de limiar, eixos do radar e o resumo do
lote. Transformadas padrão do librosa exibidas como imagem (Mel, STFT, CQT,
cromograma, fase) são calculadas pela própria apresentação.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Eixos do radar de explicabilidade, na ordem em que `radar_axes` os mede.
RADAR_AXES = (
    "Spectral",
    "Cepstral",
    "Temporal",
    "Prosodic",
    "Phase",
    "Formant",
    "HNR",
    "Jitter/Shimmer",
)

# Normalização POR EIXO, contra a faixa plausível de cada grandeza. As oito
# medidas têm unidades diferentes — CV adimensional, unidades cepstrais,
# radianos, razão de energia — e o radar dividia todas pelo MÁXIMO GLOBAL. Com
# o desvio de MFCC na casa de 10-50 e o desvio de ZCR na de 0,01, o eixo de
# MFCC ficava sempre em 1,0 e os demais colapsavam para perto de zero: o
# gráfico tinha a mesma forma para qualquer áudio.
RADAR_RANGES = (
    1.0,  # centroide espectral (CV)
    60.0,  # desvio de MFCC (unidades cepstrais)
    0.15,  # desvio de ZCR
    2.0,  # RMS (CV)
    np.pi,  # descontinuidade de fase (rad)
    0.5,  # estabilidade de formante (CV)
    1.0,  # NHR (0 = harmônico, 1 = ruído)
    0.02,  # RAP (2% já é jitter severo)
)


@dataclass(frozen=True)
class JitterShimmer:
    """Perturbação local entre quadros vozeados consecutivos, em %."""

    jitter_times: np.ndarray
    jitter: np.ndarray
    shimmer_times: np.ndarray
    shimmer: np.ndarray

    @property
    def jitter_mean(self) -> float:
        return np.mean(self.jitter)

    @property
    def shimmer_mean(self) -> float:
        return np.mean(self.shimmer)


@dataclass(frozen=True)
class PitchAnalysis:
    """Contorno de F0 (pyin) com estatísticas e flutuação de vibrato.

    ``f0_mean``/``f0_std`` são ``None`` sem quadro vozeado; ``f0_detrended``
    e ``vibrato_extent`` só existem com mais de 10 quadros vozeados.
    """

    times: np.ndarray
    f0: np.ndarray
    voiced_probs: np.ndarray
    voiced_mask: np.ndarray
    f0_mean: Optional[float]
    f0_std: Optional[float]
    f0_detrended: Optional[np.ndarray]
    vibrato_extent: Optional[float]


@dataclass(frozen=True)
class ThresholdSweep:
    """Precisão, revocação e F1 por limiar de decisão."""

    thresholds: np.ndarray
    precision: List[float]
    recall: List[float]
    f1: List[float]
    best_threshold: float


def linear_cepstrum(
    y: np.ndarray, n_filters: int = 64, n_coeffs: int = 20
) -> np.ndarray:
    """LFCC: banco de filtros LINEAR (faixas uniformes) + DCT, por quadro."""
    import librosa
    from scipy.fft import dct

    S_linear = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    S_power = S_linear**2
    n_fft_bins = S_power.shape[0]
    filterbank = np.zeros((n_filters, n_fft_bins))
    bin_width = n_fft_bins // n_filters
    for i in range(n_filters):
        start = i * bin_width
        end = min(start + bin_width, n_fft_bins)
        filterbank[i, start:end] = 1.0
    lfcc_spec = filterbank @ S_power
    lfcc_spec = librosa.power_to_db(lfcc_spec + 1e-10, ref=np.max)
    return dct(lfcc_spec, type=2, axis=0, norm="ortho")[:n_coeffs, :]


def formant_tracks(
    y: np.ndarray, sr: int
) -> Tuple[List[float], Dict[str, List[float]]]:
    """Trilhas F1-F4 por LPC em quadros de 25 ms (salto de 10 ms).

    Quadro sem raiz suficiente (ou em que a LPC falha) vira ``nan``.
    """
    import librosa

    frame_len = int(0.025 * sr)  # 25ms frames
    hop = int(0.010 * sr)  # 10ms hop
    lpc_order = 2 + int(sr / 1000)

    tracks = {f"F{i+1}": [] for i in range(4)}
    times = []

    for start in range(0, len(y) - frame_len, hop):
        frame = y[start : start + frame_len]
        frame = frame * np.hamming(len(frame))

        # LPC via autocorrelation
        try:
            a = librosa.lpc(frame, order=lpc_order)
            roots = np.roots(a)
            roots = roots[np.imag(roots) >= 0]
            angles = np.angle(roots)
            freqs = sorted(angles * (sr / (2 * np.pi)))
            freqs = [f for f in freqs if 90 < f < sr / 2]

            for i in range(4):
                if i < len(freqs):
                    tracks[f"F{i+1}"].append(freqs[i])
                else:
                    tracks[f"F{i+1}"].append(np.nan)
            times.append((start + frame_len / 2) / sr)
        except Exception:
            for i in range(4):
                tracks[f"F{i+1}"].append(np.nan)
            times.append((start + frame_len / 2) / sr)

    return times, tracks


def hnr_contour(y: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray]:
    """HNR por autocorrelação, quadro a quadro (25 ms, salto de 10 ms), em dB.

    Devolve ``(tempos, hnr)``; quadro sem pico de periodicidade vale 0 dB.
    """
    frame_len = int(0.025 * sr)
    hop = int(0.010 * sr)

    hnr_values = []
    times = []

    for start in range(0, len(y) - frame_len, hop):
        frame = y[start : start + frame_len]
        # Autocorrelation-based HNR
        try:
            autocorr = np.correlate(frame, frame, mode="full")
            autocorr = autocorr[len(autocorr) // 2 :]
            if autocorr[0] > 0:
                autocorr = autocorr / autocorr[0]
                # Find first peak after lag 0
                min_lag = int(sr / 500)  # 500 Hz max
                max_lag = int(sr / 50)  # 50 Hz min
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
        times.append((start + frame_len / 2) / sr)

    return np.array(times), np.array(hnr_values)


def _pyin(y: np.ndarray, sr: int):
    import librosa

    f0, voiced_flag, voiced_probs = librosa.pyin(
        y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"), sr=sr
    )
    return f0, voiced_probs, librosa.times_like(f0, sr=sr)


def jitter_shimmer(y: np.ndarray, sr: int) -> Optional[JitterShimmer]:
    """Jitter e shimmer locais sobre os quadros vozeados do pyin.

    ``None`` com menos de 3 quadros vozeados — não há o que comparar.
    """
    f0, _voiced_probs, times = _pyin(y, sr)

    # Voiced frames only
    voiced_idx = ~np.isnan(f0)
    f0_voiced = f0[voiced_idx]
    times_voiced = times[voiced_idx]

    if len(f0_voiced) < 3:
        return None

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

    return JitterShimmer(
        jitter_times=times_voiced[1:],
        jitter=jitter_local,
        shimmer_times=times_voiced[1 : len(shimmer_local) + 1],
        shimmer=shimmer_local,
    )


def pitch_analysis(y: np.ndarray, sr: int) -> PitchAnalysis:
    """F0 por pyin, média/desvio dos quadros vozeados e flutuação de vibrato."""
    f0, voiced_probs, times = _pyin(y, sr)
    voiced_mask = ~np.isnan(f0)

    f0_voiced = f0[voiced_mask]
    f0_mean = f0_std = None
    if len(f0_voiced) > 0:
        f0_mean = np.mean(f0_voiced)
        f0_std = np.std(f0_voiced)

    f0_detrended = vibrato_extent = None
    if len(f0_voiced) > 10:
        # Compute vibrato rate via short-term F0 fluctuation
        f0_interp = np.interp(times, times[voiced_mask], f0_voiced)
        f0_detrended = f0_interp - np.convolve(f0_interp, np.ones(20) / 20, mode="same")
        vibrato_extent = np.std(f0_detrended)

    return PitchAnalysis(
        times=times,
        f0=f0,
        voiced_probs=voiced_probs,
        voiced_mask=voiced_mask,
        f0_mean=f0_mean,
        f0_std=f0_std,
        f0_detrended=f0_detrended,
        vibrato_extent=vibrato_extent,
    )


def det_curve(
    y_true: np.ndarray, y_scores: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, float]:
    """``(fpr, fnr, eer)``: curva DET e o ponto em que FPR = FNR."""
    from sklearn.metrics import roc_curve

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

    return fpr, fnr, float(eer)


def threshold_sweep(y_true: np.ndarray, y_scores: np.ndarray) -> ThresholdSweep:
    """Precisão, revocação e F1 de 0,05 a 0,95; o melhor limiar maximiza F1."""
    from sklearn.metrics import f1_score, precision_score, recall_score

    thresholds = np.arange(0.05, 0.96, 0.01)
    precisions, recalls, f1s = [], [], []

    for t in thresholds:
        y_pred = (y_scores >= t).astype(int)
        precisions.append(precision_score(y_true, y_pred, zero_division=0))
        recalls.append(recall_score(y_true, y_pred, zero_division=0))
        f1s.append(f1_score(y_true, y_pred, zero_division=0))

    return ThresholdSweep(
        thresholds=thresholds,
        precision=precisions,
        recall=recalls,
        f1=f1s,
        best_threshold=thresholds[np.argmax(f1s)],
    )


def formant_stability(y: np.ndarray, sr: int) -> float:
    """Dispersão relativa de F1 ao longo do sinal, via LPC quadro a quadro.

    Voz sintetizada tende a apresentar trajetória de formante mais suave que a
    fala natural; a dispersão relativa (desvio sobre média) é uma medida
    adimensional disso, comparável entre áudios de durações diferentes.

    Devolve 0.0 quando não há quadros suficientes — nunca uma constante
    inventada, que era o comportamento anterior deste eixo do radar.
    """
    try:
        import librosa

        ordem = 2 + sr // 1000  # regra usual: 2 + fs/1000 coeficientes
        tamanho, salto = 1024, 512
        primeiros: list[float] = []
        for inicio in range(0, max(len(y) - tamanho, 0), salto):
            quadro = y[inicio : inicio + tamanho]
            if np.max(np.abs(quadro)) < 1e-4:  # quadro em silêncio
                continue
            coef = librosa.lpc(quadro * np.hanning(len(quadro)), order=ordem)
            raizes = np.roots(coef)
            raizes = raizes[np.imag(raizes) > 0]
            if raizes.size == 0:
                continue
            freqs = np.sort(np.angle(raizes) * (sr / (2 * np.pi)))
            # F1: primeiro pico acima de 90 Hz (abaixo disso é F0/ruído DC)
            acima = freqs[freqs > 90]
            if acima.size:
                primeiros.append(float(acima[0]))

        if len(primeiros) < 3:
            return 0.0
        media = float(np.mean(primeiros))
        return float(np.std(primeiros) / media) if media > 0 else 0.0
    except Exception as exc:  # noqa: BLE001 — eixo do radar, não pode derrubar
        logger.warning("Estabilidade de formante indisponível: %s", exc)
        return 0.0


def radar_axes(y: np.ndarray, sr: int) -> np.ndarray:
    """Os oito eixos de ``RADAR_AXES``, cada um em [0, 1] pela sua faixa."""
    import librosa

    from app.domain.features.extractors.voice_quality.components.noise import (
        compute_nhr,
    )
    from app.domain.features.extractors.voice_quality.components.perturbation import (  # noqa: E501
        compute_rap,
    )

    valores = []
    # Spectral centroid variability
    sc = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    valores.append(float(np.std(sc) / (np.mean(sc) + 1e-10)))

    # MFCC variability
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    valores.append(float(np.mean(np.std(mfcc, axis=1))))

    # ZCR variability
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    valores.append(float(np.std(zcr)))

    # RMS variability
    rms = librosa.feature.rms(y=y)[0]
    valores.append(float(np.std(rms) / (np.mean(rms) + 1e-10)))

    # Descontinuidade de fase.
    #
    # `np.angle` devolve fase em (-pi, pi]. Sem desdobrar, a diferença entre
    # quadros mede o SALTO DE WRAP, não a descontinuidade: medido, um tom puro
    # de 440 Hz dava 1,13 rad e ruído branco 2,09 rad — a métrica não zerava
    # nem para o sinal mais limpo possível. Com `unwrap` no eixo temporal, o
    # tom puro cai para 0,75.
    D = librosa.stft(y)
    phase = np.unwrap(np.angle(D), axis=1)
    phase_diff = np.diff(phase, axis=1)
    valores.append(float(np.mean(np.abs(phase_diff))))

    # Formante, HNR e jitter: MEDIDOS.
    #
    # Até 2026-07-28 estes três eixos eram as constantes 0.5, 0.4 e 0.3 —
    # marcadas como "placeholder" no código, mas exibidas no radar com os
    # rótulos "Formant", "HNR" e "Jitter/Shimmer". Três dos oito eixos de um
    # gráfico FORENSE não variavam com o áudio analisado.
    f0_contorno = librosa.yin(y, fmin=60, fmax=400, sr=sr)
    f0_contorno = np.nan_to_num(f0_contorno, nan=0.0)

    # Estabilidade de formante: dispersão relativa do primeiro formante
    # estimado por LPC quadro a quadro.
    valores.append(formant_stability(y, sr))
    # Harmonicidade: NHR (0 = puro harmônico, ~1 = puro ruído)
    valores.append(float(compute_nhr(y, f0_contorno, sr)))
    # Perturbação de período (RAP), por trecho vozeado
    valores.append(float(compute_rap(y, f0_contorno, sr)))

    return np.clip(np.array(valores) / np.array(RADAR_RANGES), 0.0, 1.0)


def batch_report(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Dados agregados de um lote de detecções, para exportação de relatório."""
    if not results:
        return {"total": 0, "summary": "Sem resultados"}

    n_fake = sum(1 for r in results if r.get("is_fake", False))
    n_real = len(results) - n_fake
    confidences = [r.get("confidence", 0) for r in results]

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
            1 for r in results if r.get("is_fake") and r.get("confidence", 0) > 0.8
        ),
        "low_confidence_detections": sum(
            1 for r in results if 0.4 < r.get("confidence", 0) < 0.6
        ),
        "per_file": [
            {
                "filename": r.get("filename", "unknown"),
                "is_fake": r.get("is_fake", False),
                "confidence": r.get("confidence", 0),
                "model": r.get("model_name", "unknown"),
            }
            for r in results
        ],
    }
