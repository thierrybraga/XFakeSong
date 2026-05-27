"""Smoke test integrado: usa FeaturePreparer + Predictor reais (módulos do app).

Diferente de test_inference_pipeline.py que chama o modelo diretamente, este teste:
1. Constrói um ModelInfo manual com modelo Keras criado via factory
2. Chama FeaturePreparer.prepare_input(audio_data, model_info, arch_info)
3. Chama Predictor.predict(model_info, features)
4. Valida que o resultado tem p_fake, p_real, is_deepfake, confidence

Cobre as 12 arquiteturas TensorFlow.

Uso standalone:
    python tests/smoke/test_inference_integrated.py
    python tests/smoke/test_inference_integrated.py --models AASIST,Conformer

Uso via pytest:
    pytest -m smoke tests/smoke/test_inference_integrated.py
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

# Fake packages para evitar app/__init__ chain (sqlalchemy/etc)
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

ALL_MODELS = [
    "AASIST", "RawGAT-ST", "RawNet2", "WavLM", "HuBERT",
    "Sonic Sleuth", "EfficientNet-LSTM", "MultiscaleCNN",
    "SpectrogramTransformer", "Conformer", "Hybrid CNN-Transformer", "Ensemble",
]


def make_audio(kind: str = "real") -> np.ndarray:
    t = np.linspace(0, DURATION, SAMPLE_RATE * DURATION, dtype=np.float32)
    if kind == "real":
        return 0.5 * np.sin(2 * np.pi * 440 * t)
    rng = np.random.default_rng(7)
    return (
        0.1 * np.sin(2 * np.pi * 880 * t)
        + 0.4 * rng.standard_normal(t.shape).astype(np.float32)
    )


def test_arch_integrated(arch: str) -> dict:
    import importlib
    import traceback

    factory_mod = importlib.import_module("app.domain.models.architectures.factory")

    # Libera stubs falsos para carregar os módulos reais de core/services
    for k in list(sys.modules.keys()):
        if (
            k.startswith("app.core")
            or k.startswith("app.domain.services.detection")
            or k == "app.domain.services"
            or k.startswith("app.domain.models")
        ) and isinstance(sys.modules[k], types.ModuleType) \
                and not hasattr(sys.modules[k], "__file__"):
            del sys.modules[k]

    audio_pre_mod = importlib.import_module("app.domain.services.detection.audio_preprocessing")

    try:
        from app.core.interfaces.audio import AudioData  # type: ignore
    except Exception:
        from dataclasses import dataclass, field

        @dataclass
        class AudioData:  # type: ignore
            samples: np.ndarray
            sample_rate: int
            duration: float
            channels: int = 1
            metadata: dict = field(default_factory=dict)

    try:
        from app.domain.services.detection.model_loader import ModelInfo
    except Exception:
        class ModelInfo:  # type: ignore
            def __init__(self, **kw):
                for k, v in kw.items():
                    setattr(self, k, v)

    fp_mod = importlib.import_module("app.domain.services.detection.feature_preparer")
    pred_mod = importlib.import_module("app.domain.services.detection.predictor")

    spec = factory_mod.architecture_factory_registry.get_architecture_info(arch)
    input_type = spec.input_requirements.get("input_type", "spectrogram")
    if input_type == "raw_audio":
        input_shape = (SAMPLE_RATE * DURATION, 1)
    else:
        t_frames = int(np.ceil((SAMPLE_RATE * DURATION) / audio_pre_mod.DEFAULT_HOP))
        input_shape = (t_frames, audio_pre_mod.DEFAULT_N_MELS, 1)

    model = factory_mod.create_model_by_name(arch, input_shape=input_shape, num_classes=2)

    model_info = ModelInfo(
        name=f"test_{arch.lower().replace(' ', '_').replace('-', '_')}",
        architecture=arch, model=model, scaler=None,
        input_shape=input_shape, model_type="tensorflow",
        input_contract=None, temperature=1.0,
        jit_predict_fn=None, eer_threshold=None,
    )

    class _MockFS:
        def extract_features(self, *a, **kw): return None
        def extract_segmented_features(self, *a, **kw): return None

    fp = fp_mod.FeaturePreparer(_MockFS())
    predictor = pred_mod.Predictor()

    out = {}
    for kind in ["real", "fake"]:
        audio_samples = make_audio(kind)
        try:
            ad = AudioData(samples=audio_samples, sample_rate=SAMPLE_RATE, duration=float(DURATION))
        except Exception as e:
            return {"arch": arch, "ok": False, "error": f"AudioData ctor: {e}\n{traceback.format_exc()[-500:]}"}

        try:
            prep = fp.prepare_input(ad, model_info, spec)
        except Exception as e:
            return {"arch": arch, "ok": False, "error": f"prep raised: {e}\n{traceback.format_exc()[-500:]}"}

        if prep["status"] != "ok":
            return {"arch": arch, "ok": False, "error": f"prep failed [{kind}]: {prep.get('error')}"}

        features = prep["features"]
        if features.ndim != len(input_shape):
            return {"arch": arch, "ok": False, "error": f"Shape mismatch: {features.shape} vs dims={len(input_shape)}"}

        try:
            pred_res = predictor.predict(model_info, features)
        except Exception as e:
            return {"arch": arch, "ok": False, "error": f"predict raised: {e}\n{traceback.format_exc()[-500:]}"}

        if pred_res.status.value != "success":
            return {"arch": arch, "ok": False, "error": f"predict failed [{kind}]: {pred_res.errors}"}

        d = pred_res.data
        ok = (
            isinstance(d.get("is_deepfake"), bool)
            and all(k in d for k in ("p_fake", "p_real", "confidence"))
            and 0.0 <= d["p_fake"] <= 1.0 + 1e-3
            and 0.0 <= d["p_real"] <= 1.0 + 1e-3
            and abs(d["p_fake"] + d["p_real"] - 1.0) < 1e-2
        )
        out[kind] = {
            "ok": ok, "is_deepfake": d.get("is_deepfake"),
            "p_fake": d.get("p_fake"), "p_real": d.get("p_real"),
            "confidence": d.get("confidence"), "features_shape": tuple(features.shape),
        }

    return {"arch": arch, "ok": all(v["ok"] for v in out.values()), "input_type": input_type, "results": out}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", default="", help="CSV de modelos (vazio = todos)")
    args = parser.parse_args(argv)

    targets = [m.strip() for m in args.models.split(",")] if args.models else ALL_MODELS

    print("=" * 80)
    print("INTEGRATED inference smoke test (FeaturePreparer + Predictor)")
    print("=" * 80)

    results = []
    for arch in targets:
        print(f"\n>>> {arch}")
        try:
            t0 = time.time()
            r = test_arch_integrated(arch)
            r["elapsed"] = time.time() - t0
            results.append(r)
            if r["ok"]:
                print(f"  [OK] input_type={r['input_type']} ({r['elapsed']:.1f}s)")
                for kind, v in r["results"].items():
                    print(f"    [{kind:5s}] is_fake={v['is_deepfake']} p_fake={v['p_fake']:.4f} p_real={v['p_real']:.4f} feats={v['features_shape']}")
            else:
                print(f"  [FAIL] {r.get('error', 'unknown')}")
        except Exception as e:
            import traceback as _tb
            print(f"  [FAIL] {type(e).__name__}: {str(e)[:200]}")
            _tb.print_exc()
            results.append({"arch": arch, "ok": False, "error": str(e)})

    ok_count = sum(1 for r in results if r.get("ok"))
    print("\n" + "=" * 80)
    print(f"SUMÁRIO: {ok_count}/{len(results)} arquiteturas funcionais end-to-end")
    for r in results:
        print(f"  [{'OK  ' if r.get('ok') else 'FAIL'}] {r.get('arch', '?')}")
    return 0 if ok_count == len(results) else 1


# ── pytest integration ────────────────────────────────────────────────────────
import pytest  # noqa: E402


@pytest.mark.smoke
def test_smoke_inference_integrated() -> None:
    """FeaturePreparer + Predictor funcionam end-to-end para as 12 arquiteturas."""
    rc = main(argv=[])
    assert rc == 0, "Uma ou mais arquiteturas falharam — veja saída acima"


# ── standalone ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sys.exit(main())
