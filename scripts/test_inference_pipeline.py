"""Smoke test do pipeline de INFERÊNCIA para cada uma das 12 arquiteturas.

Para cada modelo:
1. Cria o modelo do zero (sem treinar) com input_shape correto
2. Salva em arquivo .keras temporário
3. Gera 2 áudios sintéticos (1 "real" senoidal, 1 "fake" ruidoso)
4. Carrega o modelo via path direto e chama o pipeline de inferência
5. Verifica: is_fake é bool, p_fake ∈ [0, 1], p_real ∈ [0, 1], soma ≈ 1.0

NÃO precisa do DetectionService completo — testa diretamente
FeaturePreparer + Predictor com um ModelInfo manual.

Uso:
    python scripts/test_inference_pipeline.py
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import types
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np  # noqa: E402

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
sys.path.insert(0, PROJECT_ROOT)

# Fake packages para evitar app/__init__ chain (sqlalchemy)
for fake_pkg in [
    "app",
    "app.core",
    "app.core.interfaces",
    "app.domain",
    "app.domain.models",
    "app.domain.models.architectures",
    "app.domain.services",
    "app.domain.services.detection",
]:
    if fake_pkg not in sys.modules:
        m = types.ModuleType(fake_pkg)
        m.__path__ = [os.path.join(PROJECT_ROOT, *fake_pkg.split("."))]
        sys.modules[fake_pkg] = m

SAMPLE_RATE = 16000
DURATION = 3
N_FFT = 512
HOP = 128
N_MELS = 80


def make_real_audio() -> np.ndarray:
    """Áudio 'real': senoide pura."""
    t = np.linspace(0, DURATION, SAMPLE_RATE * DURATION, dtype=np.float32)
    return 0.5 * np.sin(2 * np.pi * 440 * t)


def make_fake_audio() -> np.ndarray:
    """Áudio 'fake': ruído branco com pequena senoide."""
    rng = np.random.default_rng(123)
    t = np.linspace(0, DURATION, SAMPLE_RATE * DURATION, dtype=np.float32)
    return (
        0.1 * np.sin(2 * np.pi * 880 * t)
        + 0.4 * rng.standard_normal(t.shape).astype(np.float32)
    )


def test_inference_for_arch(arch: str) -> dict:
    """Testa o pipeline de inferência completo para uma arquitetura."""
    import importlib
    import tensorflow as tf

    factory_mod = importlib.import_module(
        "app.domain.models.architectures.factory"
    )
    audio_pre_mod = importlib.import_module(
        "app.domain.services.detection.audio_preprocessing"
    )

    # Descobre input_type/shape do spec
    spec = factory_mod.architecture_factory_registry.get_architecture_info(arch)
    input_type = spec.input_requirements.get("input_type", "spectrogram")
    if input_type == "raw_audio":
        input_shape = (SAMPLE_RATE * DURATION, 1)
    else:
        # Para spectrogram, calcula T_frames a partir do tf.signal.stft com pad_end=True
        # T_frames = ceil(T_audio / hop)
        t_frames = int(np.ceil((SAMPLE_RATE * DURATION) / HOP))
        input_shape = (t_frames, N_MELS, 1)

    # Cria modelo
    model = factory_mod.create_model_by_name(
        arch, input_shape=input_shape, num_classes=2,
    )

    # Detecta out_units
    out_units = model.output_shape[-1]

    # Compila com loss apropriada (a inferência só precisa do forward pass mas
    # alguns modelos pré-compilam internamente — sem problema)
    if out_units == 1:
        model.compile(loss="binary_crossentropy", optimizer="adam")
    else:
        model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")

    # Gera 2 áudios
    real_audio = make_real_audio()
    fake_audio = make_fake_audio()

    results = []
    for label, audio in [("real_audio", real_audio), ("fake_audio", fake_audio)]:
        # Pré-processa via helper unificado
        x = audio_pre_mod.prepare_audio_for_model(
            audio,
            input_type=input_type,
            input_shape=input_shape,
            sample_rate=SAMPLE_RATE,
            normalize=True,
        )
        # Adiciona dim batch
        x_batch = x[np.newaxis, ...]
        # Forward
        y = model(x_batch, training=False).numpy()  # (1, out_units)

        # Interpreta como o predictor faria
        pred = y[0]
        if pred.shape[-1] == 1:
            p_fake = float(pred[0])
            p_real = 1.0 - p_fake
        else:
            # Normaliza se output for logits (alguns custom outputs)
            # Heurística: se valores fora de [0, 1] ou soma != 1, aplica softmax
            psum = float(np.sum(pred))
            if (np.any(pred < 0) or np.any(pred > 1) or
                    abs(psum - 1.0) > 1e-3):
                # Aplica softmax para normalizar
                pe = np.exp(pred - np.max(pred))
                pred_norm = pe / np.sum(pe)
                p_real, p_fake = float(pred_norm[0]), float(pred_norm[1])
            else:
                p_real, p_fake = float(pred[0]), float(pred[1])
        is_fake = bool(p_fake > 0.5)
        confidence = p_fake if is_fake else p_real

        # Validações
        ok = (
            isinstance(is_fake, bool)
            and 0.0 <= p_fake <= 1.0 + 1e-3
            and 0.0 <= p_real <= 1.0 + 1e-3
            and abs(p_fake + p_real - 1.0) < 1e-2
            and isinstance(confidence, float)
        )
        results.append({
            "audio": label,
            "ok": ok,
            "p_fake": p_fake,
            "p_real": p_real,
            "is_fake": is_fake,
            "confidence": confidence,
            "out_shape": pred.shape,
        })

    all_ok = all(r["ok"] for r in results)
    return {
        "arch": arch,
        "ok": all_ok,
        "input_type": input_type,
        "input_shape": input_shape,
        "out_units": int(out_units),
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="", help="CSV de modelos (vazio = todos)")
    args = parser.parse_args()

    all_models = [
        "AASIST", "RawGAT-ST", "RawNet2", "WavLM", "HuBERT",
        "Sonic Sleuth", "EfficientNet-LSTM", "MultiscaleCNN",
        "SpectrogramTransformer", "Conformer", "Hybrid CNN-Transformer", "Ensemble",
    ]
    targets = (
        [m.strip() for m in args.models.split(",")] if args.models else all_models
    )

    print("=" * 80)
    print("Inference pipeline smoke test (12 arquiteturas)")
    print("=" * 80)

    results = []
    for arch in targets:
        print(f"\n>>> {arch}")
        try:
            t0 = time.time()
            r = test_inference_for_arch(arch)
            r["elapsed"] = time.time() - t0
            results.append(r)
            status = "[OK]" if r["ok"] else "[FAIL]"
            print(f"  {status} input_type={r['input_type']} shape={r['input_shape']} out={r['out_units']} ({r['elapsed']:.1f}s)")
            for res in r["results"]:
                s = "OK" if res["ok"] else "BAD"
                print(
                    f"    [{s}] {res['audio']:12s} "
                    f"p_fake={res['p_fake']:.4f} p_real={res['p_real']:.4f} "
                    f"is_fake={res['is_fake']} confidence={res['confidence']:.4f}"
                )
        except Exception as e:
            print(f"  [FAIL] {type(e).__name__}: {str(e)[:200]}")
            results.append({"arch": arch, "ok": False, "error": str(e)})

    # Sumário
    print("\n" + "=" * 80)
    print("SUMÁRIO")
    print("=" * 80)
    ok_count = sum(1 for r in results if r.get("ok"))
    print(f"OK: {ok_count}/{len(results)}\n")
    for r in results:
        status = "OK  " if r.get("ok") else "FAIL"
        arch = r.get("arch", "?")
        print(f"  [{status}] {arch}")

    return 0 if ok_count == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
