#!/usr/bin/env python3
"""
robustness_test.py — Teste de robustez sob ruido aditivo (AWGN).

Avalia todos os modelos treinados em SNR = 10, 20, 30 dB.
Gera Tabela 5 do TCC.

Uso:
  python scripts/robustness_test.py                          # todos os modelos
  python scripts/robustness_test.py --model multiscale_cnn   # um modelo
  python scripts/robustness_test.py --snr 5 10 20            # SNRs customizados
"""

import argparse
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

SAMPLE_RATE    = 16_000
SPLITS_DIR     = BASE_DIR / "app" / "datasets" / "splits"
MODELS_DIR     = BASE_DIR / "results" / "models"
OUTPUT_PATH    = BASE_DIR / "results" / "robustness_results.json"

# Default SNR levels tested (dB)
SNR_LEVELS_DEFAULT = [10, 20, 30]

# Arquiteturas a testar: nome → (modulo, variante, num_classes)
# num_classes=2 para AASIST (AMSoftmax), 1 para as demais (sigmoid binario)
ARCH_REGISTRY = {
    "multiscale_cnn": (
        "app.domain.models.architectures.multiscale_cnn",
        "multiscale_cnn", 1,
    ),
    "ensemble_adaptive": (
        "app.domain.models.architectures.ensemble",
        "ensemble_adaptive", 1,
    ),
    "rawnet2": (
        "app.domain.models.architectures.rawnet2",
        "rawnet2", 1,
    ),
    "efficientnet_lstm": (
        "app.domain.models.architectures.efficientnet_lstm",
        "efficientnet_lstm", 1,
    ),
    "aasist": (
        "app.domain.models.architectures.aasist",
        "aasist", 2,
    ),
}

# Arquiteturas que exigem audio longo (raw-audio com GPU)
RAW_AUDIO_ARCHS = {"efficientnet_lstm", "aasist", "rawnet2"}


# ---------------------------------------------------------------------------
# GPU detection — mirrors setup_gpu() in train_advanced.py
# ---------------------------------------------------------------------------

def detect_gpu() -> bool:
    """Retorna True se GPU disponivel (detectada antes de qualquer op TF)."""
    import tensorflow as tf
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        return True
    return False


# ---------------------------------------------------------------------------
# Utilidades de audio
# ---------------------------------------------------------------------------

def add_awgn(audio: np.ndarray, snr_db: float) -> np.ndarray:
    """Adiciona ruido branco gaussiano para atingir SNR alvo (dB)."""
    signal_power = np.mean(audio ** 2) + 1e-10
    noise_power  = signal_power / (10 ** (snr_db / 10.0))
    noise = np.random.randn(len(audio)).astype(np.float32) * np.sqrt(noise_power)
    return np.clip(audio + noise, -1.0, 1.0)


def load_split(split: str, max_samples: int) -> tuple:
    """Carrega split (test/val) com VAD+AGC."""
    import librosa
    from app.core.utils.silero_vad import preprocess_audio

    split_dir = SPLITS_DIR / split
    X, y = [], []
    for label, subdir in [(0, "real"), (1, "fake")]:
        class_dir = split_dir / subdir
        if not class_dir.exists():
            logger.warning(f"Diretorio nao encontrado: {class_dir}")
            continue
        files = sorted(class_dir.glob("*.wav"))
        for fpath in files:
            try:
                audio, _ = librosa.load(str(fpath), sr=SAMPLE_RATE, mono=True)
                audio = audio.astype(np.float32)
                audio, _ = preprocess_audio(audio, apply_vad=True, apply_gain=True)
                if len(audio) >= max_samples:
                    audio = audio[:max_samples]
                else:
                    audio = np.pad(audio, (0, max_samples - len(audio)))
                X.append(audio)
                y.append(label)
            except Exception as e:
                logger.debug(f"Erro {fpath.name}: {e}")

    if not X:
        return np.array([]), np.array([])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


# ---------------------------------------------------------------------------
# Metricas — suporta saida binaria (sigmoid) e 2-classes (AMSoftmax + softmax)
# ---------------------------------------------------------------------------

def compute_metrics(model, X: np.ndarray, y: np.ndarray) -> dict:
    """Acuracia, EER e AUC-ROC.

    Detecta automaticamente o tipo de saida:
    - (N,)  ou (N,1): probabilidade binaria (sigmoid)
    - (N,2)         : AMSoftmax 2-classes — usa coluna 1 (P_fake)
    """
    from sklearn.metrics import roc_auc_score, roc_curve
    from scipy.interpolate import interp1d
    from scipy.optimize import brentq

    preds_raw = model.predict(X, verbose=0)

    if preds_raw.ndim == 2 and preds_raw.shape[1] == 2:
        scores = preds_raw[:, 1].astype(float)   # P(fake)
    else:
        scores = preds_raw.ravel().astype(float)

    predicted = (scores >= 0.5).astype(int)
    accuracy  = float(np.mean(predicted == y))

    eer = 1.0 - accuracy
    auc = 0.5
    try:
        auc = float(roc_auc_score(y, scores))
        fpr, tpr, thresholds = roc_curve(y, scores)
        fnr = 1.0 - tpr
        # Interpolacao precisa do EER
        fpr_fn = interp1d(thresholds[::-1], fpr[::-1], bounds_error=False, fill_value=(0, 1))
        fnr_fn = interp1d(thresholds[::-1], fnr[::-1], bounds_error=False, fill_value=(0, 1))
        t_min, t_max = float(thresholds.min()), float(thresholds.max())
        if t_min < t_max:
            tau = brentq(lambda t: float(fpr_fn(t)) - float(fnr_fn(t)), t_min, t_max)
            eer = float(fpr_fn(tau))
        else:
            idx = int(np.argmin(np.abs(fpr - fnr)))
            eer = float((fpr[idx] + fnr[idx]) / 2.0)
    except Exception:
        pass

    return {
        "accuracy_pct": round(accuracy * 100, 2),
        "eer_pct":       round(eer * 100, 2),
        "auc_roc":       round(auc, 3),
    }


# ---------------------------------------------------------------------------
# Carregamento de modelo salvo
# ---------------------------------------------------------------------------

def load_model(arch_name: str, gpu_available: bool):
    """Carrega o melhor checkpoint do modelo treinado.

    Reconstroi a arquitetura (para garantir que layers customizadas sao
    registradas) e carrega os pesos do .h5 salvo pelo ModelCheckpoint.
    """
    import importlib
    import tensorflow as tf

    model_path = MODELS_DIR / f"{arch_name}_best.h5"
    if not model_path.exists():
        logger.warning(f"Checkpoint nao encontrado: {model_path}")
        return None

    module_path, variant, num_classes = ARCH_REGISTRY[arch_name]
    is_raw = arch_name in RAW_AUDIO_ARCHS
    audio_samples = (
        48_000 if (is_raw and gpu_available) else 16_000
    )

    try:
        mod = importlib.import_module(module_path)
        model = mod.create_model(
            input_shape=(audio_samples,),
            num_classes=num_classes,
            architecture=variant,
        )
        model.load_weights(str(model_path))
        logger.info(
            f"  Modelo {arch_name} carregado "
            f"({audio_samples // SAMPLE_RATE}s, "
            f"{model.count_params():,} params)"
        )
        return model, audio_samples
    except Exception as e:
        logger.error(f"Falha ao carregar {arch_name}: {e}")
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Teste de robustez sob AWGN para todos os modelos treinados"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        choices=list(ARCH_REGISTRY.keys()),
        help="Arquitetura especifica (default: todos os modelos com checkpoint)",
    )
    parser.add_argument(
        "--snr", type=float, nargs="+", default=SNR_LEVELS_DEFAULT,
        help="Niveis de SNR em dB (default: 10 20 30)",
    )
    parser.add_argument(
        "--split", type=str, default="test",
        choices=["test", "val"],
        help="Split a usar para avaliacao (default: test)",
    )
    args = parser.parse_args()

    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")

    # Registrar layers customizadas (necessario para load_weights)
    logger.info("Registrando layers customizadas...")
    try:
        import app.domain.models.architectures.multiscale_cnn  as _mcnn  # noqa
        import app.domain.models.architectures.ensemble         as _ens   # noqa
        import app.domain.models.architectures.rawnet2          as _rn2   # noqa
        import app.domain.models.architectures.efficientnet_lstm as _elstm # noqa
        import app.domain.models.architectures.aasist           as _aas   # noqa
        logger.info("  Layers registradas com sucesso.")
    except Exception as e:
        logger.warning(f"  Falha parcial ao registrar layers: {e}")

    # Detectar GPU
    gpu_available = detect_gpu()
    logger.info(f"Hardware: {'GPU' if gpu_available else 'CPU'}")

    snr_levels = sorted(args.snr, reverse=True)   # ex: [30, 20, 10]

    # Arquiteturas a testar
    archs_to_run = [args.model] if args.model else list(ARCH_REGISTRY.keys())

    # Filtrar: testar apenas modelos cujo checkpoint existe
    archs_available = []
    for arch in archs_to_run:
        if (MODELS_DIR / f"{arch}_best.h5").exists():
            archs_available.append(arch)
        else:
            logger.info(f"  {arch}: sem checkpoint — pulando")

    if not archs_available:
        logger.error(
            f"Nenhum checkpoint encontrado em {MODELS_DIR}. "
            "Execute train_advanced.py primeiro."
        )
        sys.exit(1)

    logger.info(f"\nModelos a testar: {archs_available}")
    logger.info(f"SNR levels (dB) : {snr_levels}")

    all_results = {}

    for arch_name in archs_available:
        logger.info(f"\n{'='*55}")
        logger.info(f"AVALIANDO: {arch_name.upper()}")
        logger.info(f"{'='*55}")

        loaded = load_model(arch_name, gpu_available)
        if loaded is None:
            continue
        model, audio_samples = loaded

        # Carregar split com comprimento de audio correto para este modelo
        X_clean, y_test = load_split(args.split, audio_samples)
        if len(X_clean) == 0:
            logger.warning(f"Split '{args.split}' vazio para {arch_name}. Pulando.")
            continue

        logger.info(
            f"  Split '{args.split}': {len(X_clean)} amostras "
            f"({int((y_test==0).sum())} real + {int((y_test==1).sum())} fake)"
        )

        arch_results = {}

        # Avaliacao limpa (referencia sem ruido)
        clean_m = compute_metrics(model, X_clean, y_test)
        arch_results["clean"] = clean_m
        logger.info(
            f"  [limpo ] acc={clean_m['accuracy_pct']:5.1f}%  "
            f"EER={clean_m['eer_pct']:5.1f}%  AUC={clean_m['auc_roc']:.3f}"
        )

        # Avaliacao com AWGN por nivel de SNR
        for snr in snr_levels:
            X_noisy = np.stack([add_awgn(x, snr) for x in X_clean])
            m = compute_metrics(model, X_noisy, y_test)
            arch_results[f"snr_{int(snr)}db"] = m
            logger.info(
                f"  [SNR{snr:4.0f}dB] acc={m['accuracy_pct']:5.1f}%  "
                f"EER={m['eer_pct']:5.1f}%  AUC={m['auc_roc']:.3f}"
            )

        all_results[arch_name] = arch_results

    if not all_results:
        logger.error("Nenhum resultado gerado.")
        sys.exit(1)

    # Salvar JSON
    output = {
        "experiment": {
            "split": args.split,
            "snr_levels_db": snr_levels,
            "noise_type": "AWGN (white gaussian noise)",
            "gpu_available": gpu_available,
        },
        "results": all_results,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    logger.info(f"\nResultados salvos em: {OUTPUT_PATH}")

    # ── Tabela 5 — Acuracia (%) ──────────────────────────────────────────
    snr_cols = [int(s) for s in snr_levels]
    header = f"{'Arquitetura':<22} {'Limpo':>8}" + "".join(f" {'SNR'+str(s)+'dB':>9}" for s in snr_cols)

    print(f"\n{'='*65}")
    print("TABELA 5 — ROBUSTEZ AWGN: Acuracia (%)")
    print("="*65)
    print(header)
    print("-"*65)
    for arch, r in all_results.items():
        row = f"{arch:<22} {r['clean']['accuracy_pct']:>7.1f}%"
        for s in snr_cols:
            row += f" {r[f'snr_{s}db']['accuracy_pct']:>8.1f}%"
        print(row)

    print(f"\n{'='*65}")
    print("TABELA 5 — ROBUSTEZ AWGN: EER (%)")
    print("="*65)
    print(header.replace("Acuracia", "EER    "))
    print("-"*65)
    for arch, r in all_results.items():
        row = f"{arch:<22} {r['clean']['eer_pct']:>7.1f}%"
        for s in snr_cols:
            row += f" {r[f'snr_{s}db']['eer_pct']:>8.1f}%"
        print(row)

    print(f"\n{'='*65}")
    print("TABELA 5 — ROBUSTEZ AWGN: AUC-ROC")
    print("="*65)
    print(header.replace("Acuracia", "AUC-ROC"))
    print("-"*65)
    for arch, r in all_results.items():
        row = f"{arch:<22} {r['clean']['auc_roc']:>8.3f}"
        for s in snr_cols:
            row += f" {r[f'snr_{s}db']['auc_roc']:>9.3f}"
        print(row)


if __name__ == "__main__":
    main()
