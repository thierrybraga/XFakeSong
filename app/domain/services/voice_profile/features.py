"""Vetor de features dos perfis de voz e amostras fake sintéticas.

O mesmo vetor (144 dimensões: MFCC e deltas, espectrais, contraste, F0,
chroma e tonnetz) serve ao treino e à verificação; os fakes saem de
perturbações das próprias amostras do perfil (pitch, vocoder, compressão).
"""

import logging
from pathlib import Path
from typing import List, Optional

import librosa
import numpy as np

logger = logging.getLogger(__name__)

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

# Feature configuration padrão
DEFAULT_FEATURE_CONFIG = {
    "n_mfcc": 13,
    "sr_target": 16000,
    "feature_set": "full",  # full | compact
    "features": [
        "mfcc_stats",  # 13 * 4 = 52
        "delta_mfcc",  # 13 * 2 = 26
        "spectral_stats",  # 6 * 2 = 12  (centroid, bandwidth, rolloff, flatness, zcr, rms)
        "spectral_contrast",  # 7 * 2 = 14
        "f0_stats",  # 4
        "chroma_stats",  # 12 * 2 = 24
        "tonnetz_stats",  # 6 * 2 = 12
    ],
    "total_dim": 144,
}


def extract_profile_features(y: np.ndarray, sr: int) -> Optional[np.ndarray]:
    """Extrai vetor de features para treinamento/inferência de perfil.

    Retorna vetor de 144 dimensões com estatísticas agregadas
    (``DEFAULT_FEATURE_CONFIG["total_dim"]``).
    """
    try:
        features = []

        # MFCCs (13 coefs x 4 stats = 52)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        features.extend(np.mean(mfcc, axis=1))
        features.extend(np.std(mfcc, axis=1))
        features.extend(np.max(mfcc, axis=1) - np.min(mfcc, axis=1))
        features.extend(np.median(mfcc, axis=1))

        # Delta MFCCs (13 x 2 stats = 26)
        delta_mfcc = librosa.feature.delta(mfcc)
        features.extend(np.mean(delta_mfcc, axis=1))
        features.extend(np.std(delta_mfcc, axis=1))

        # Spectral features (7 features x 2 stats = 14)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        spectral_flatness = librosa.feature.spectral_flatness(y=y)
        zcr = librosa.feature.zero_crossing_rate(y)
        rms = librosa.feature.rms(y=y)

        for feat in [
            spectral_centroid,
            spectral_bandwidth,
            spectral_rolloff,
            spectral_flatness,
            zcr,
            rms,
        ]:
            features.extend([np.mean(feat), np.std(feat)])

        # Spectral contrast (7 bands x 2 stats = 14)
        features.extend(np.mean(spectral_contrast, axis=1))
        features.extend(np.std(spectral_contrast, axis=1))

        # F0 / Pitch (4 features)
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"), sr=sr
        )
        f0_valid = f0[~np.isnan(f0)] if f0 is not None else np.array([0])
        if len(f0_valid) == 0:
            f0_valid = np.array([0])
        features.extend(
            [
                np.mean(f0_valid),
                np.std(f0_valid),
                np.max(f0_valid) - np.min(f0_valid),
                np.mean(voiced_probs) if voiced_probs is not None else 0.0,
            ]
        )

        # Chroma (12 x 2 stats = 24)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        features.extend(np.mean(chroma, axis=1))
        features.extend(np.std(chroma, axis=1))

        # Tonnetz (6 x 2 = 12)
        try:
            tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
            features.extend(np.mean(tonnetz, axis=1))
            features.extend(np.std(tonnetz, axis=1))
        except Exception:
            features.extend([0.0] * 12)

        return np.array(features, dtype=np.float32)

    except Exception as e:
        logger.warning(f"Erro na extração de features: {e}")
        return None


def generate_synthetic_fakes(
    real_features: List[np.ndarray],
    samples_dir: Path,
    sr_target: int,
) -> List[np.ndarray]:
    """Gera features de amostras fake via perturbações realistas.

    Inclui perturbações que simulam artefatos de deepfake:
    - Pitch shift (conversão de voz)
    - Ruído gaussiano (artefatos de vocoder)
    - Time stretch (distorções temporais)
    - Formant warp (pitch + speed combo)
    - Low-pass filter (perda de alta frequência de vocoders)
    - Quantization noise (artefatos de compressão neural)
    - Spectral smoothing (suavização típica de TTS)
    """
    import scipy.signal as signal

    fake_features = []
    audio_files = sorted(
        f for f in samples_dir.iterdir() if f.suffix.lower() in AUDIO_EXTENSIONS
    )

    for audio_file in audio_files:
        try:
            y, sr = librosa.load(str(audio_file), sr=sr_target, mono=True)

            perturbations = []

            # 1. Pitch shift +3 semitons (conversão de voz leve)
            try:
                y_p1 = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=3)
                perturbations.append(y_p1)
            except Exception:
                pass

            # 2. Pitch shift -3 semitons
            try:
                y_p2 = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=-3)
                perturbations.append(y_p2)
            except Exception:
                pass

            # 3. Ruído gaussiano (artefatos de vocoder)
            noise = np.random.randn(len(y)) * 0.03
            perturbations.append(y + noise)

            # 4. Time stretch rápido
            try:
                y_fast = librosa.effects.time_stretch(y=y, rate=1.2)
                perturbations.append(y_fast)
            except Exception:
                pass

            # 5. Formant warp (pitch +5 + slow)
            try:
                y_fw = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=5)
                y_fw = librosa.effects.time_stretch(y=y_fw, rate=0.85)
                perturbations.append(y_fw)
            except Exception:
                pass

            # 6. Low-pass filter (simula vocoder que perde altas freq.)
            try:
                nyquist = sr / 2
                cutoff = 3500 / nyquist  # Corta acima de 3.5kHz
                if 0 < cutoff < 1:
                    b, a = signal.butter(4, cutoff, btype="low")
                    y_lp = signal.filtfilt(b, a, y).astype(np.float32)
                    perturbations.append(y_lp)
            except Exception:
                pass

            # 7. Quantization noise (simula artefatos de compressão neural)
            levels = 256
            y_quant = np.round(y * levels) / levels
            noise_q = np.random.randn(len(y)) * 0.01
            perturbations.append(y_quant + noise_q)

            # Extrair features de cada perturbação
            for y_pert in perturbations:
                feats = extract_profile_features(y_pert, sr)
                if feats is not None:
                    fake_features.append(feats)

        except Exception as e:
            logger.warning(f"Erro ao gerar fake de {audio_file.name}: {e}")

    # Se houver poucas fakes, interpolar features com ruído
    while len(fake_features) < len(real_features):
        idx = np.random.randint(0, len(real_features))
        noise_feat = real_features[idx] + np.random.randn(
            len(real_features[idx])
        ) * 0.3 * np.std(real_features[idx])
        fake_features.append(noise_feat)

    return fake_features
