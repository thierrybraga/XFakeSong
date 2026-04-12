#!/usr/bin/env python3
"""
robustness_test.py — Teste de robustez sob ruido aditivo (AWGN).

Avalia MultiscaleCNN e Ensemble Adaptativo em SNR = 10, 20, 30 dB.
Gera Tabela 5 do TCC.

Uso:
  python scripts/robustness_test.py
"""

import json
import logging
import os
import sys
import warnings
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

SAMPLE_RATE = 16_000
MAX_AUDIO_SAMPLES = 16_000
SPLITS_DIR = BASE_DIR / "app" / "datasets" / "splits"
MODELS_DIR = BASE_DIR / "results" / "models"
OUTPUT_PATH = BASE_DIR / "results" / "robustness_results.json"

SNR_LEVELS = [10, 20, 30]  # dB

MODELS_TO_TEST = {
    "multiscale_cnn":    MODELS_DIR / "multiscale_cnn_best.h5",
    "ensemble_adaptive": MODELS_DIR / "ensemble_adaptive_best.h5",
}


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def add_awgn(audio: np.ndarray, snr_db: float) -> np.ndarray:
    """Adiciona ruido branco gaussiano para atingir SNR alvo (dB)."""
    signal_power = np.mean(audio ** 2) + 1e-10
    noise_power  = signal_power / (10 ** (snr_db / 10.0))
    noise = np.random.randn(len(audio)).astype(np.float32) * np.sqrt(noise_power)
    return np.clip(audio + noise, -1.0, 1.0)


def load_split(split: str = "test") -> tuple:
    """Carrega split (test) com VAD/AGC simplificado."""
    import librosa
    from app.core.utils.silero_vad import preprocess_audio

    split_dir = SPLITS_DIR / split
    X, y = [], []
    for label, subdir in [(0, "real"), (1, "fake")]:
        class_dir = split_dir / subdir
        if not class_dir.exists():
            continue
        for fpath in sorted(class_dir.glob("*.wav")):
            try:
                audio, _ = librosa.load(str(fpath), sr=SAMPLE_RATE, mono=True)
                audio = audio.astype(np.float32)
                audio, _ = preprocess_audio(audio, apply_vad=True, apply_gain=True)
                if len(audio) >= MAX_AUDIO_SAMPLES:
                    audio = audio[:MAX_AUDIO_SAMPLES]
                else:
                    audio = np.pad(audio, (0, MAX_AUDIO_SAMPLES - len(audio)))
                X.append(audio)
                y.append(label)
            except Exception as e:
                logger.debug(f"Erro {fpath.name}: {e}")

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def compute_metrics(model, X: np.ndarray, y: np.ndarray) -> dict:
    """Acuracia, EER e AUC-ROC."""
    from scipy.optimize import brentq
    from scipy.interpolate import interp1d
    from sklearn.metrics import roc_curve, roc_auc_score

    preds = model.predict(X, verbose=0).flatten()
    predicted = (preds >= 0.5).astype(int)
    accuracy = float(np.mean(predicted == y))

    eer = 1.0 - accuracy
    auc = 0.5
    try:
        auc = float(roc_auc_score(y, preds))
        fpr, tpr, thresholds = roc_curve(y, preds)
        fnr = 1 - tpr
        fpr_fn = interp1d(thresholds[::-1], fpr[::-1], bounds_error=False, fill_value=(0, 1))
        fnr_fn = interp1d(thresholds[::-1], fnr[::-1], bounds_error=False, fill_value=(0, 1))
        t_min, t_max = float(thresholds.min()), float(thresholds.max())
        tau = brentq(lambda t: float(fpr_fn(t)) - float(fnr_fn(t)), t_min, t_max)
        eer = float(fpr_fn(tau))
    except Exception:
        pass

    return {
        "accuracy_pct": round(accuracy * 100, 2),
        "eer_pct":       round(eer * 100, 2),
        "auc_roc":       round(auc, 3),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")

    # Habilitar deserializacao de Lambda/funcoes locais e registrar layers
    import keras
    keras.config.enable_unsafe_deserialization()

    logger.info("Registrando layers customizadas...")
    try:
        import app.domain.models.architectures.multiscale_cnn as _mcnn  # noqa
        import app.domain.models.architectures.ensemble as _ens          # noqa
        logger.info("  Layers registradas com sucesso.")
    except Exception as e:
        logger.warning(f"  Falha ao registrar layers: {e}")

    logger.info("Carregando split de teste...")
    X_clean, y_test = load_split("test")
    logger.info(f"  {len(X_clean)} amostras | real={int((y_test==0).sum())} fake={int((y_test==1).sum())}")

    results = {}

    arch_factory = {
        "multiscale_cnn": ("app.domain.models.architectures.multiscale_cnn", "multiscale_cnn"),
        "ensemble_adaptive": ("app.domain.models.architectures.ensemble", "ensemble_adaptive"),
    }

    for arch_name, model_path in MODELS_TO_TEST.items():
        if not model_path.exists():
            logger.warning(f"Modelo nao encontrado: {model_path}. Pulando.")
            continue

        logger.info(f"\nRecriando arquitetura e carregando pesos: {arch_name}")
        try:
            import importlib
            mod_path, variant = arch_factory[arch_name]
            mod = importlib.import_module(mod_path)
            model = mod.create_model(input_shape=(MAX_AUDIO_SAMPLES,), num_classes=1, architecture=variant)
            model.load_weights(str(model_path))
            logger.info(f"  Pesos carregados com sucesso.")
        except Exception as e:
            logger.error(f"Falha ao carregar {arch_name}: {e}")
            continue

        arch_results = {}

        # Avaliacao limpa (referencia)
        clean_metrics = compute_metrics(model, X_clean, y_test)
        arch_results["clean"] = clean_metrics
        logger.info(f"  [clean] Acuracia={clean_metrics['accuracy_pct']:.1f}%  EER={clean_metrics['eer_pct']:.1f}%  AUC={clean_metrics['auc_roc']:.3f}")

        # Avaliacao com ruido por SNR
        for snr in SNR_LEVELS:
            X_noisy = np.stack([add_awgn(x, snr) for x in X_clean])
            metrics = compute_metrics(model, X_noisy, y_test)
            arch_results[f"snr_{snr}db"] = metrics
            logger.info(f"  [SNR {snr:2d}dB] Acuracia={metrics['accuracy_pct']:.1f}%  EER={metrics['eer_pct']:.1f}%  AUC={metrics['auc_roc']:.3f}")

        results[arch_name] = arch_results

    # Salvar resultados
    output = {
        "experiment": {
            "test_samples": int(len(X_clean)),
            "snr_levels_db": SNR_LEVELS,
            "noise_type": "AWGN (white gaussian noise)",
            "audio_input_s": MAX_AUDIO_SAMPLES / SAMPLE_RATE,
        },
        "results": results,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    logger.info(f"\nResultados salvos em: {OUTPUT_PATH}")

    # Imprimir Tabela 5
    print("\n=== TABELA 5 — ROBUSTEZ (Acuracia %) ===\n")
    header = f"{'Arquitetura':<25} {'Limpo':>8} {'SNR30':>8} {'SNR20':>8} {'SNR10':>8}"
    print(header)
    print("-" * 60)
    for arch, r in results.items():
        row = f"{arch:<25}"
        row += f" {r['clean']['accuracy_pct']:>7.1f}%"
        for snr in [30, 20, 10]:
            k = f"snr_{snr}db"
            row += f" {r[k]['accuracy_pct']:>7.1f}%"
        print(row)

    print("\n=== TABELA 5 — ROBUSTEZ (EER %) ===\n")
    print(header.replace("Acuracia", "EER    "))
    print("-" * 60)
    for arch, r in results.items():
        row = f"{arch:<25}"
        row += f" {r['clean']['eer_pct']:>7.1f}%"
        for snr in [30, 20, 10]:
            k = f"snr_{snr}db"
            row += f" {r[k]['eer_pct']:>7.1f}%"
        print(row)


if __name__ == "__main__":
    main()
