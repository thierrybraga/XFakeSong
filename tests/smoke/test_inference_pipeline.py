"""Smoke test do pipeline de INFERÊNCIA para cada uma das 12 arquiteturas.

Para cada modelo:
1. Cria o modelo do zero (sem treinar) com input_shape correto
2. Gera 2 áudios sintéticos (1 "real" senoidal, 1 "fake" ruidoso)
3. Pré-processa via audio_preprocessing.prepare_audio_for_model
4. Forward pass direto no modelo
5. Verifica: is_fake é bool, p_fake ∈ [0, 1], p_real ∈ [0, 1], soma ≈ 1.0

Uso standalone:
    python tests/smoke/test_inference_pipeline.py
    python tests/smoke/test_inference_pipeline.py --models AASIST,Conformer

Uso via pytest:
    pytest -m smoke tests/smoke/test_inference_pipeline.py
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import types

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np  # noqa: E402

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(THIS_DIR))  # tests/smoke/ → tests/ → raiz
sys.path.insert(0, PROJECT_ROOT)

# Fake packages para evitar app/__init__ chain (sqlalchemy)
for _fake_pkg in [
    "app", "app.core", "app.core.interfaces",
    "app.domain", "app.domain.models", "app.domain.models.architectures",
    "app.domain.services", "app.domain.services.detection",
]:
    if _fake_pkg not in sys.modules:
        _m = types.ModuleType(_fake_pkg)
        _m.__path__ = [os.path.join(PROJECT_ROOT, *_fake_pkg.split("."))]
        sys.modules[_fake_pkg] = _m

SAMPLE_RATE = 16000
DURATION = 3
N_FFT = 512
HOP = 128
N_MELS = 80

ALL_MODELS = [
    "AASIST", "RawGAT-ST", "RawNet2", "WavLM", "HuBERT",
    "Sonic Sleuth", "EfficientNet-LSTM", "MultiscaleCNN",
    "SpectrogramTransformer", "Conformer", "Hybrid CNN-Transformer", "Ensemble",
]


def make_real_audio() -> np.ndarray:
    t = np.linspace(0, DURATION, SAMPLE_RATE * DURATION, dtype=np.float32)
    return 0.5 * np.sin(2 * np.pi * 440 * t)


def make_fake_audio() -> np.ndarray:
    rng = np.random.default_rng(123)
    t = np.linspace(0, DURATION, SAMPLE_RATE * DURATION, dtype=np.float32)
    return (
        0.1 * np.sin(2 * np.pi * 880 * t)
        + 0.4 * rng.standard_normal(t.shape).astype(np.float32)
    )


def test_inference_for_arch(arch: str) -> dict:
    """Testa o pipeline de inferência direto para uma arquitetura."""
    import importlib
    factory_mod = importlib.import_module("app.domain.models.architectures.factory")
    audio_pre_mod = importlib.import_module("app.domain.services.detection.audio_preprocessing")

    spec = factory_mod.architecture_factory_registry.get_architecture_info(arch)
    input_type = spec.input_requirements.get("input_type", "spectrogram")
    if input_type == "raw_audio":
        input_shape = (SAMPLE_RATE * DURATION, 1)
    else:
        t_frames = int(np.ceil((SAMPLE_RATE * DURATION) / HOP))
        input_shape = (t_frames, N_MELS, 1)

    model = factory_mod.create_model_by_name(arch, input_shape=input_shape, num_classes=2)
    out_units = model.output_shape[-1]

    results = []
    for label, audio in [("real_audio", make_real_audio()), ("fake_audio", make_fake_audio())]:
        x = audio_pre_mod.prepare_audio_for_model(
            audio, input_type=input_type, input_shape=input_shape,
            sample_rate=SAMPLE_RATE, normalize=True,
        )
        y = model(x[np.newaxis, ...], training=False).numpy()
        pred = y[0]
        if pred.shape[-1] == 1:
            p_fake = float(pred[0])
            p_real = 1.0 - p_fake
        else:
            psum = float(np.sum(pred))
            if np.any(pred < 0) or np.any(pred > 1) or abs(psum - 1.0) > 1e-3:
                pe = np.exp(pred - np.max(pred))
                pred = pe / np.sum(pe)
            p_real, p_fake = float(pred[0]), float(pred[1])
        is_fake = bool(p_fake > 0.5)
        ok = (
            isinstance(is_fake, bool)
            and 0.0 <= p_fake <= 1.0 + 1e-3
            and 0.0 <= p_real <= 1.0 + 1e-3
            and abs(p_fake + p_real - 1.0) < 1e-2
        )
        results.append({"audio": label, "ok": ok, "p_fake": p_fake, "p_real": p_real, "is_fake": is_fake})

    return {
        "arch": arch, "ok": all(r["ok"] for r in results),
        "input_type": input_type, "input_shape": input_shape,
        "out_units": int(out_units), "results": results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", default="", help="CSV de modelos (vazio = todos)")
    args = parser.parse_args(argv)

    targets = [m.strip() for m in args.models.split(",")] if args.models else ALL_MODELS

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
                print(f"    [{s}] {res['audio']:12s} p_fake={res['p_fake']:.4f} p_real={res['p_real']:.4f} is_fake={res['is_fake']}")
        except Exception as e:
            print(f"  [FAIL] {type(e).__name__}: {str(e)[:200]}")
            results.append({"arch": arch, "ok": False, "error": str(e)})

    ok_count = sum(1 for r in results if r.get("ok"))
    print("\n" + "=" * 80)
    print(f"SUMÁRIO: OK {ok_count}/{len(results)}")
    for r in results:
        print(f"  [{'OK  ' if r.get('ok') else 'FAIL'}] {r.get('arch')}")
    return 0 if ok_count == len(results) else 1


# ── pytest integration ────────────────────────────────────────────────────────
import pytest  # noqa: E402


@pytest.mark.smoke
def test_smoke_inference_pipeline() -> None:
    """Pipeline de inferência direta funciona para todas as 12 arquiteturas."""
    rc = main(argv=[])
    assert rc == 0, "Uma ou mais arquiteturas falharam na inferência — veja saída acima"


# ── standalone ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sys.exit(main())
