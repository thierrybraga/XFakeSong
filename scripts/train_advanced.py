#!/usr/bin/env python3
"""
train_advanced.py — Pipeline de Treinamento para TCC UFSJ 2026.

Treina cada arquitetura nas splits do dataset PT-BR e registra metricas
reais para substituir os valores inventados do TCC (Tabela 3).

Uso:
  python scripts/train_advanced.py                        # todas as arquiteturas
  python scripts/train_advanced.py --model conformer      # uma arquitetura
  python scripts/train_advanced.py --epochs 100           # epocas maximas
  python scripts/train_advanced.py --quick                # 20 epocas, debug rapido
"""

import argparse
import json
import logging
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Disable oneDNN to avoid MaxPool MKL memory bug
warnings.filterwarnings("ignore")

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("TrainAdvanced")

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

SAMPLE_RATE = 16_000
MAX_AUDIO_SAMPLES_CPU = 16_000   # 1s @ 16kHz — CPU training
MAX_AUDIO_SAMPLES_GPU = 48_000   # 3s @ 16kHz — GPU training (raw-audio models)
MAX_AUDIO_SAMPLES     = MAX_AUDIO_SAMPLES_CPU   # updated by setup_gpu()
RESULTS_DIR = BASE_DIR / "results"

# Arquiteturas que exigem GPU para convergir (raw-audio / heavy backbones)
RAW_AUDIO_ARCHS = {"efficientnet_lstm", "aasist", "rawnet2"}

# Arquiteturas a treinar (nome -> modulo, variante)
ARCHITECTURES = {
    "efficientnet_lstm": ("app.domain.models.architectures.efficientnet_lstm", "efficientnet_lstm"),
    "multiscale_cnn":    ("app.domain.models.architectures.multiscale_cnn",    "multiscale_cnn"),
    "aasist":            ("app.domain.models.architectures.aasist",            "aasist"),
    "rawnet2":           ("app.domain.models.architectures.rawnet2",           "rawnet2"),
    "ensemble_adaptive": ("app.domain.models.architectures.ensemble",          "ensemble_adaptive"),
}

# GPU state — set by setup_gpu()
GPU_AVAILABLE: bool = False
STRATEGY = None


def setup_gpu() -> "tf.distribute.Strategy":
    """Detect GPU, enable memory growth and mixed precision.

    Returns a tf.distribute.Strategy suitable for the available hardware:
    - MirroredStrategy  (multi-GPU)
    - OneDeviceStrategy('/GPU:0')  (single GPU)
    - default Strategy  (CPU-only)
    """
    global GPU_AVAILABLE, STRATEGY, MAX_AUDIO_SAMPLES
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            # Mixed precision: float16 compute, float32 weights → ~2× speedup on
            # Tensor Cores (Volta / Turing / Ampere).  Output layers must be cast
            # to float32 explicitly (handled in each architecture).
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
            logger.info(
                f"GPU detectada: {len(gpus)} dispositivo(s). "
                "Mixed precision (float16) ativado."
            )
            GPU_AVAILABLE = True
            MAX_AUDIO_SAMPLES = MAX_AUDIO_SAMPLES_GPU
            STRATEGY = (
                tf.distribute.MirroredStrategy()
                if len(gpus) > 1
                else tf.distribute.OneDeviceStrategy("/GPU:0")
            )
        except Exception as exc:
            logger.warning(f"Erro ao configurar GPU: {exc}")
            STRATEGY = tf.distribute.get_strategy()
    else:
        logger.info("Nenhuma GPU detectada. CPU sera usada.")
        STRATEGY = tf.distribute.get_strategy()

    return STRATEGY


# ---------------------------------------------------------------------------
# Carregamento de dados
# ---------------------------------------------------------------------------

def load_audio(path: Path, max_samples: int = None) -> np.ndarray:
    """Carrega WAV, aplica VAD+AGC e padeia/trunca para max_samples.

    max_samples defaults to the global MAX_AUDIO_SAMPLES (16 k on CPU,
    80 k on GPU) so callers can override when needed.
    """
    if max_samples is None:
        max_samples = MAX_AUDIO_SAMPLES

    import librosa
    from app.core.utils.silero_vad import preprocess_audio

    y, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
    y = y.astype(np.float32)
    y, _ = preprocess_audio(y, apply_vad=True, apply_gain=True)

    if len(y) >= max_samples:
        y = y[:max_samples]
    else:
        y = np.pad(y, (0, max_samples - len(y)))
    return y


def load_split(split_dir: Path, max_per_class: int = None) -> tuple:
    """Carrega uma split (train/val/test) de app/datasets/splits/{split}/."""
    X, y = [], []
    for label, subdir in [(0, "real"), (1, "fake")]:
        class_dir = split_dir / subdir
        if not class_dir.exists():
            logger.warning(f"Diretorio nao encontrado: {class_dir}")
            continue
        files = sorted(class_dir.glob("*.wav"))
        if max_per_class:
            files = files[:max_per_class]
        for i, f in enumerate(files):
            try:
                audio = load_audio(f)
                X.append(audio)
                y.append(label)
            except Exception as e:
                logger.debug(f"Erro ao carregar {f.name}: {e}")
            if (i + 1) % 200 == 0:
                logger.info(f"    [{subdir}] {i+1}/{len(files)} carregados...")

    if not X:
        return np.array([]), np.array([])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def prepare_splits(splits_dir: Path, datasets_dir: Path, min_per_class: int = 100) -> bool:
    """Garante que splits existem com amostras suficientes.

    Ordem de prioridade:
    1. Splits já existem com >= min_per_class amostras por classe → usa diretamente
       (inclui dados baixados pela aba Dataset do Gradio)
    2. Dados brutos em real/ + fake/ existem → regenera splits via preprocess_dataset.py
    3. Dados insuficientes → loga instrução e retorna False
    """
    import subprocess

    train_real = len(list((splits_dir / "train" / "real").glob("*.wav")))
    train_fake = len(list((splits_dir / "train" / "fake").glob("*.wav")))

    if train_real >= min_per_class and train_fake >= min_per_class:
        val_real  = len(list((splits_dir / "val"  / "real").glob("*.wav")))
        val_fake  = len(list((splits_dir / "val"  / "fake").glob("*.wav")))
        test_real = len(list((splits_dir / "test" / "real").glob("*.wav")))
        test_fake = len(list((splits_dir / "test" / "fake").glob("*.wav")))
        logger.info(
            f"Splits encontrados (train {train_real}+{train_fake} | "
            f"val {val_real}+{val_fake} | test {test_real}+{test_fake}). "
            "Usando dados existentes."
        )
        return True

    # Splits insuficientes — verificar dados brutos
    real_raw = len(list((datasets_dir / "real").glob("*.wav")))
    fake_raw = len(list((datasets_dir / "fake").glob("*.wav")))

    if real_raw >= min_per_class and fake_raw >= min_per_class:
        logger.info(
            f"Dados brutos encontrados ({real_raw} real + {fake_raw} fake). "
            "Recriando splits..."
        )
        result = subprocess.run(
            [sys.executable, str(BASE_DIR / "scripts" / "preprocess_dataset.py"),
             "--full", "--train-ratio", "0.70", "--val-ratio", "0.15", "--test-ratio", "0.15"],
            cwd=str(BASE_DIR),
        )
        return result.returncode == 0

    # Dados insuficientes
    logger.error(
        f"Dados insuficientes: {real_raw} real + {fake_raw} fake "
        f"(minimo {min_per_class} por classe). "
        "Opcoes:\n"
        "  1. Use a aba 'Dataset' no Gradio para baixar dados\n"
        "  2. Execute: python scripts/build_dataset.py --target 500 --skip-real-cv\n"
        "  3. Execute este script com: --build-dataset"
    )
    return False


# ---------------------------------------------------------------------------
# Treinamento
# ---------------------------------------------------------------------------

def build_model(arch_name: str, input_shape: tuple, num_classes: int = 1):
    """Importa e cria modelo pela arquitetura."""
    import importlib
    module_path, variant = ARCHITECTURES[arch_name]
    mod = importlib.import_module(module_path)
    # AASIST uses AMSoftmax which needs 2 weight vectors for binary real/fake
    # detection — with num_classes=1 the cosine loss is flat and cannot learn.
    nc = 2 if arch_name == "aasist" else num_classes
    return mod.create_model(
        input_shape=input_shape,
        num_classes=nc,
        architecture=variant,
    )


def get_callbacks(model_path: Path, patience: int = 10, lr_patience: int = 5):
    """Callbacks alinhados com TCC Secao 5.2."""
    import tensorflow as tf
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=lr_patience,
            min_lr=1e-7,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_path),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=0,
        ),
    ]


def compute_metrics(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Calcula accuracy, EER, AUC-ROC conforme TCC Secao 6.2."""
    import tensorflow as tf
    from sklearn.metrics import roc_auc_score, roc_curve

    preds_raw = model.predict(X_test, verbose=0)
    # 2-class softmax output (e.g. AASIST): take P(fake) = column 1.
    # Binary sigmoid output: flatten to 1-D score vector.
    if preds_raw.ndim == 2 and preds_raw.shape[1] == 2:
        scores = preds_raw[:, 1].astype(float)
    else:
        scores = preds_raw.ravel().astype(float)
    preds = (scores > 0.5).astype(int)

    accuracy = float(np.mean(preds == y_test))

    try:
        auc = float(roc_auc_score(y_test, scores))
    except Exception:
        auc = 0.0

    # EER (Eq. 30 do TCC): tau tal que FAR(tau) = FRR(tau)
    try:
        fpr, tpr, thresholds = roc_curve(y_test, scores)
        fnr = 1.0 - tpr
        diff = np.abs(fpr - fnr)
        idx = int(np.argmin(diff))
        eer = float((fpr[idx] + fnr[idx]) / 2.0)
        # Tentar interpolacao mais precisa
        try:
            from scipy.interpolate import interp1d
            from scipy.optimize import brentq
            fpr_fn = interp1d(thresholds[::-1], fpr[::-1], bounds_error=False, fill_value=(1, 0))
            fnr_fn = interp1d(thresholds[::-1], fnr[::-1], bounds_error=False, fill_value=(0, 1))
            t_min, t_max = float(thresholds.min()), float(thresholds.max())
            tau = brentq(lambda t: float(fpr_fn(t)) - float(fnr_fn(t)), t_min, t_max)
            eer = float(fpr_fn(tau))
        except Exception:
            pass
    except Exception:
        eer = 1.0 - accuracy

    return {
        "accuracy": round(accuracy * 100, 2),
        "eer": round(eer * 100, 2),
        "auc_roc": round(auc, 3),
    }


def measure_latency(model, n_samples: int = 30, audio_duration_s: float = 10.0) -> float:
    """Mede latencia media de inferencia para audio de 10s (ms)."""
    # Use MAX_AUDIO_SAMPLES to match model's expected input shape
    dummy = np.random.randn(1, MAX_AUDIO_SAMPLES).astype(np.float32)
    # Warmup
    for _ in range(3):
        model.predict(dummy, verbose=0)
    times = []
    for _ in range(n_samples):
        t0 = time.perf_counter()
        model.predict(dummy, verbose=0)
        times.append((time.perf_counter() - t0) * 1000)
    return round(float(np.mean(times)), 1)


def augment_batch(X: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Augmentation conforme TCC Sec 6.1: pitch ±2 semitones, time 0.9-1.1x."""
    import librosa
    out = []
    for audio in X:
        if np.random.rand() < 0.5:
            n_steps = np.random.uniform(-2, 2)
            audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)
        if np.random.rand() < 0.5:
            rate = np.random.uniform(0.9, 1.1)
            stretched = librosa.effects.time_stretch(audio, rate=rate)
            if len(stretched) >= MAX_AUDIO_SAMPLES:
                stretched = stretched[:MAX_AUDIO_SAMPLES]
            else:
                stretched = np.pad(stretched, (0, MAX_AUDIO_SAMPLES - len(stretched)))
            audio = stretched
        out.append(audio[:MAX_AUDIO_SAMPLES])
    return np.array(out, dtype=np.float32)


def train_model(
    arch_name: str,
    X_train, y_train,
    X_val, y_val,
    X_test, y_test,
    epochs: int = 100,
    batch_size: int = 32,
    models_dir: Path = None,
    force_cpu: bool = False,
) -> dict:
    """Treina uma arquitetura e retorna metricas completas."""
    import tensorflow as tf

    # Raw-audio models require GPU to converge. Skip gracefully when unavailable.
    is_raw_audio = arch_name in RAW_AUDIO_ARCHS
    if is_raw_audio and not GPU_AVAILABLE and not force_cpu:
        msg = (
            f"{arch_name} nao convergiu em CPU (requer GPU + >5.000 amostras). "
            "Use --force-cpu para treinar mesmo assim ou conecte uma GPU."
        )
        logger.warning(msg)
        return {"skipped": True, "reason": msg}

    device_label = "GPU" if (is_raw_audio and GPU_AVAILABLE) else "CPU"
    logger.info(f"\n{'='*60}")
    logger.info(f"TREINANDO: {arch_name.upper()} [{device_label}]")
    logger.info(f"  Train : {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    logger.info(f"  Epochs: {epochs} | Batch: {batch_size} | Patience: 10")
    if is_raw_audio and GPU_AVAILABLE:
        logger.info(f"  Mixed precision: float16 | Audio: {MAX_AUDIO_SAMPLES/SAMPLE_RATE:.0f}s segments")
    logger.info(f"{'='*60}")

    input_shape = (X_train.shape[1],)
    model_path = (models_dir or RESULTS_DIR / "models") / f"{arch_name}_best.h5"
    model_path.parent.mkdir(parents=True, exist_ok=True)

    from app.domain.models.training.optimized_training_config import get_recommended_hyperparameters
    hp = get_recommended_hyperparameters(arch_name.replace("_", "-").title())
    lr = hp.get("learning_rate", 0.001)
    l2 = hp.get("l2_reg_strength", 0.0001)

    # Build + compile inside strategy scope (enables GPU distribution and
    # mixed-precision variable casting automatically).
    try:
        with STRATEGY.scope():
            model = build_model(arch_name, input_shape, num_classes=1)
            out_dim = model.output_shape[-1]   # 2 for AASIST, 1 for others
            if out_dim == 2:
                # AASIST: 2-class AMSoftmax + softmax output.
                # Use AdamW (paper-specified) + SparseCategoricalCrossentropy.
                model.compile(
                    optimizer=tf.keras.optimizers.AdamW(
                        learning_rate=lr, weight_decay=0.01
                    ),
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                    metrics=["accuracy"],
                )
            else:
                model.compile(
                    optimizer=tf.keras.optimizers.Adam(
                        learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-7
                    ),
                    loss=tf.keras.losses.BinaryCrossentropy(),
                    metrics=["accuracy"],
                )
    except Exception as e:
        logger.error(f"Falha ao criar {arch_name}: {e}")
        return {"error": str(e)}

    # ── class weights ────────────────────────────────────────────────────────
    n_real = int(np.sum(y_train == 0))
    n_fake = int(np.sum(y_train == 1))
    total = n_real + n_fake
    class_weight = {
        0: total / (2.0 * n_real) if n_real > 0 else 1.0,
        1: total / (2.0 * n_fake) if n_fake > 0 else 1.0,
    }

    callbacks = get_callbacks(model_path, patience=10, lr_patience=5)

    # ── tf.data pipeline ─────────────────────────────────────────────────────
    n_samples = X_train.shape[1]  # actual length after load (may be 16k or 80k)

    def make_dataset(X, y, augment=False, shuffle=False):
        ds = tf.data.Dataset.from_tensor_slices((X, y.astype(np.float32)))
        if shuffle:
            ds = ds.shuffle(len(X), seed=42)
        if augment:
            def augment_map(xb, yb):
                augmented = tf.numpy_function(
                    lambda x: augment_batch(x).astype(np.float32),
                    [xb], tf.float32,
                )
                # Restore static shape so model layers can use shape info
                augmented = tf.ensure_shape(augmented, [None, n_samples])
                return augmented, yb
            ds = ds.batch(batch_size).map(
                augment_map,
                num_parallel_calls=tf.data.AUTOTUNE,
            )
        else:
            ds = ds.batch(batch_size)
        return ds.prefetch(tf.data.AUTOTUNE)

    train_ds = make_dataset(X_train, y_train, augment=True, shuffle=True)
    val_ds   = make_dataset(X_val,   y_val,   augment=False)

    t_start = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1,
    )
    train_time = round(time.time() - t_start, 1)

    # Carregar melhor checkpoint
    if model_path.exists():
        try:
            model = tf.keras.models.load_model(str(model_path))
        except Exception:
            pass

    # Metricas no conjunto de teste
    test_metrics = compute_metrics(model, X_test, y_test)
    latency_ms   = measure_latency(model)
    params       = model.count_params()

    best_val_acc = max(history.history.get("val_accuracy", [0])) * 100
    epochs_run   = len(history.history.get("val_loss", []))

    result = {
        "architecture": arch_name,
        "device": device_label,
        "audio_length_s": round(n_samples / SAMPLE_RATE, 1),
        "params": params,
        "memory_mb": round(params * 4 / 1024**2, 1),
        "train_samples": int(len(X_train)),
        "val_samples": int(len(X_val)),
        "test_samples": int(len(X_test)),
        "epochs_run": epochs_run,
        "epochs_max": epochs,
        "train_time_s": train_time,
        "best_val_accuracy_pct": round(best_val_acc, 2),
        "test_accuracy_pct": test_metrics["accuracy"],
        "test_eer_pct": test_metrics["eer"],
        "test_auc_roc": test_metrics["auc_roc"],
        "latency_10s_ms": latency_ms,
        "hyperparams": {
            "lr": lr, "l2": l2, "batch_size": batch_size,
            "patience": 10, "lr_patience": 5,
        },
        "model_path": str(model_path),
        "timestamp": datetime.now().isoformat(),
    }

    logger.info(f"\n  Resultado {arch_name}:")
    logger.info(f"    Acuracia  : {result['test_accuracy_pct']:.1f}%")
    logger.info(f"    EER       : {result['test_eer_pct']:.1f}%")
    logger.info(f"    AUC-ROC   : {result['test_auc_roc']:.3f}")
    logger.info(f"    Latencia  : {result['latency_10s_ms']} ms")
    logger.info(f"    Epocas    : {epochs_run}/{epochs}")
    logger.info(f"    Tempo     : {train_time}s")

    return result


# ---------------------------------------------------------------------------
# Tabela 3 do TCC
# ---------------------------------------------------------------------------

def print_table3(results: dict):
    logger.info("\n" + "=" * 75)
    logger.info("TABELA 3 — Desempenho das Arquiteturas Implementadas (RESULTADOS REAIS)")
    logger.info("=" * 75)
    logger.info(
        f"  {'Arquitetura':<22} {'Acuracia':>10} {'EER':>8} {'AUC-ROC':>9} {'Latencia':>12}"
    )
    logger.info("  " + "-" * 63)
    for arch, r in results.items():
        if "error" in r:
            logger.info(f"  {arch:<22}  ERRO: {r['error'][:30]}")
            continue
        logger.info(
            f"  {arch:<22} {r['test_accuracy_pct']:>9.1f}% "
            f"{r['test_eer_pct']:>7.1f}% "
            f"{r['test_auc_roc']:>9.3f} "
            f"{r['latency_10s_ms']:>9.0f} ms"
        )
    logger.info("=" * 75)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Pipeline de treinamento para TCC UFSJ 2026"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        choices=list(ARCHITECTURES.keys()),
        help="Arquitetura especifica (default: todas em sequencia)",
    )
    parser.add_argument(
        "--epochs", type=int, default=100,
        help="Epocas maximas (default: 100, early stopping patience=10)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Batch size (default: 32)",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Modo rapido: 20 epocas, sem augmentation (debug)",
    )
    parser.add_argument(
        "--force-cpu", action="store_true",
        help=(
            "Treine modelos raw-audio mesmo sem GPU. "
            "NAO e recomendado — eles nao convergem em CPU. "
            "Util apenas para testar o pipeline."
        ),
    )
    parser.add_argument(
        "--build-dataset", action="store_true",
        help=(
            "Baixar/atualizar o dataset antes de treinar "
            "(BRSpeech-DF + Fake Voices; equivalente a build_dataset.py --skip-real-cv). "
            "Usa os mesmos dados da aba Dataset do Gradio."
        ),
    )
    parser.add_argument(
        "--min-samples", type=int, default=100,
        help="Minimo de amostras por classe no split de treino (default: 100)",
    )
    parser.add_argument(
        "--robustness", action="store_true",
        help="Executar teste de robustez (AWGN) automaticamente apos o treinamento",
    )
    parser.add_argument(
        "--output", type=str,
        default=str(RESULTS_DIR / "training_metrics.json"),
        help="Arquivo de saida JSON com metricas",
    )
    args = parser.parse_args()

    # ── GPU setup (must happen before any TF op) ──────────────────────────
    setup_gpu()

    # ── Dataset build (optional, before loading splits) ───────────────────
    if args.build_dataset:
        import subprocess as _sp
        logger.info("\n>>> ETAPA 0: BUILD DATASET (--build-dataset)")
        logger.info(
            "Baixando BRSpeech-DF + Fake Voices — mesmos dados da aba Dataset do Gradio"
        )
        ret = _sp.run(
            [sys.executable,
             str(BASE_DIR / "scripts" / "build_dataset.py"),
             "--skip-real-cv",          # usa apenas BRSpeech + FakeVoices (mais rapido)
             "--target", "1000"],        # 1000 por classe = 2000 amostras total
            cwd=str(BASE_DIR),
        ).returncode
        if ret != 0:
            logger.warning("build_dataset.py retornou erro. Tentando continuar com dados existentes.")

    if args.quick:
        args.epochs = 20
        logger.info("MODO RAPIDO: 20 epocas")

    if GPU_AVAILABLE:
        logger.info(
            f"Modelos raw-audio ({', '.join(sorted(RAW_AUDIO_ARCHS))}) "
            f"serao treinados com {MAX_AUDIO_SAMPLES // SAMPLE_RATE}s de audio em GPU."
        )
    else:
        skippable = sorted(RAW_AUDIO_ARCHS)
        logger.warning(
            f"Sem GPU: modelos {skippable} serao ignorados. "
            "Use --force-cpu para treinar mesmo assim (sem garantia de convergencia)."
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    models_dir = RESULTS_DIR / "models"
    models_dir.mkdir(exist_ok=True)

    splits_dir   = BASE_DIR / "app" / "datasets" / "splits"
    datasets_dir = BASE_DIR / "app" / "datasets"

    # Garantir splits existem com amostras suficientes
    if not prepare_splits(splits_dir, datasets_dir, min_per_class=args.min_samples):
        logger.error(
            "Nao foi possivel criar/encontrar splits suficientes. "
            "Use --build-dataset para baixar dados automaticamente."
        )
        sys.exit(1)

    # Verificar tamanho do dataset
    real_train = len(list((splits_dir / "train" / "real").glob("*.wav")))
    fake_train = len(list((splits_dir / "train" / "fake").glob("*.wav")))
    real_val   = len(list((splits_dir / "val"   / "real").glob("*.wav")))
    fake_val   = len(list((splits_dir / "val"   / "fake").glob("*.wav")))
    real_test  = len(list((splits_dir / "test"  / "real").glob("*.wav")))
    fake_test  = len(list((splits_dir / "test"  / "fake").glob("*.wav")))
    logger.info(
        f"Dataset: train {real_train}+{fake_train} | "
        f"val {real_val}+{fake_val} | test {real_test}+{fake_test} "
        f"(total={real_train+fake_train+real_val+fake_val+real_test+fake_test})"
    )

    if real_train + fake_train < 20:
        logger.error(
            "Dataset muito pequeno para treinamento serio. "
            "Execute com --build-dataset ou use a aba Dataset do Gradio."
        )
        sys.exit(1)

    if real_train + fake_train < 200:
        logger.warning(
            f"Dataset pequeno ({real_train + fake_train} amostras de treino). "
            "Para resultados definitivos, baixe mais dados via --build-dataset "
            "ou pela aba Dataset no Gradio."
        )

    # Carregar splits
    logger.info("\nCarregando splits...")
    logger.info("  [train]...")
    X_train, y_train = load_split(splits_dir / "train")
    logger.info("  [val]...")
    X_val,   y_val   = load_split(splits_dir / "val")
    logger.info("  [test]...")
    X_test,  y_test  = load_split(splits_dir / "test")

    if len(X_train) == 0:
        logger.error("Falha ao carregar dados de treino.")
        sys.exit(1)

    logger.info(
        f"\nDados carregados: train={len(X_train)} | val={len(X_val)} | test={len(X_test)}"
    )
    logger.info(
        f"  Train: {int(np.sum(y_train==0))} real + {int(np.sum(y_train==1))} fake"
    )
    logger.info(
        f"  Test : {int(np.sum(y_test==0))} real + {int(np.sum(y_test==1))} fake"
    )

    # Arquiteturas a treinar
    archs_to_run = [args.model] if args.model else list(ARCHITECTURES.keys())

    # Carregar resultados anteriores (para retomar)
    all_results = {}
    output_path = Path(args.output)
    if output_path.exists():
        try:
            with open(output_path, encoding="utf-8") as f:
                all_results = json.load(f).get("architectures", {})
            logger.info(f"Resultados anteriores carregados: {list(all_results.keys())}")
        except Exception:
            pass

    # Treinar cada arquitetura
    for arch_name in archs_to_run:
        if arch_name in all_results and "error" not in all_results[arch_name]:
            logger.info(f"\nPulando {arch_name} (ja treinado). Use --model {arch_name} para re-treinar.")
            continue

        result = train_model(
            arch_name=arch_name,
            X_train=X_train, y_train=y_train,
            X_val=X_val,     y_val=y_val,
            X_test=X_test,   y_test=y_test,
            epochs=args.epochs,
            batch_size=args.batch_size,
            models_dir=models_dir,
            force_cpu=args.force_cpu,
        )
        all_results[arch_name] = result

        # Salvar apos cada modelo (tolerante a interrupcoes)
        output = {
            "dataset": {
                "train": len(X_train),
                "val": len(X_val),
                "test": len(X_test),
                "train_real": int(np.sum(y_train == 0)),
                "train_fake": int(np.sum(y_train == 1)),
                "audio_length_s_cpu": MAX_AUDIO_SAMPLES_CPU / SAMPLE_RATE,
                "audio_length_s_gpu": MAX_AUDIO_SAMPLES_GPU / SAMPLE_RATE,
                "sample_rate": SAMPLE_RATE,
            },
            "hardware": {
                "gpu_available": GPU_AVAILABLE,
                "device": "GPU" if GPU_AVAILABLE else "CPU",
                "mixed_precision": GPU_AVAILABLE,
            },
            "training": {
                "epochs_max": args.epochs,
                "batch_size": args.batch_size,
                "early_stopping_patience": 10,
                "lr_patience": 5,
                "augmentation": "pitch +-2 semitones, time 0.9-1.1x",
                "split": "70/15/15 estratificado",
            },
            "architectures": all_results,
            "generated_at": datetime.now().isoformat(),
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2, default=str)

    # Tabela 3 final
    print_table3(all_results)

    logger.info(f"\nMetricas salvas em: {output_path}")

    # ── Robustness test (opcional) ─────────────────────────────────────────
    if args.robustness:
        import subprocess as _sp
        logger.info("\n>>> ETAPA ROBUSTEZ: Testando modelos com AWGN (SNR 10/20/30 dB)")
        _sp.run(
            [sys.executable, str(BASE_DIR / "scripts" / "robustness_test.py")],
            cwd=str(BASE_DIR),
        )
    else:
        logger.info("\nProximos passos:")
        logger.info("  python scripts/robustness_test.py   # Tabela 5 (robustez AWGN)")
        logger.info("  python scripts/run_shap_analysis.py # Analise SHAP")


if __name__ == "__main__":
    main()
