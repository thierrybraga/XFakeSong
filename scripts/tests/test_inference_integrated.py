"""Smoke test integrado: usa FeaturePreparer + Predictor reais (módulos do app).

Diferente de test_inference_pipeline.py que chamava o modelo diretamente,
este teste:
1. Constrói um ModelInfo manual com modelo Keras criado via factory
2. Chama FeaturePreparer.prepare_input(audio_data, model_info, arch_info)
3. Chama Predictor.predict_batch(model_info, [features])
4. Valida que o resultado tem p_fake, p_real, is_deepfake, confidence

Cobre os 12 arquiteturas TensorFlow.
"""

from __future__ import annotations

import argparse
import os
import sys
import types
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np  # noqa: E402

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(THIS_DIR))  # scripts/tests/ → scripts/ → raiz
sys.path.insert(0, PROJECT_ROOT)

# Fake packages para evitar app/__init__ chain (sqlalchemy/etc)
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


def make_audio(kind: str = "real") -> np.ndarray:
    """Áudio sintético: senoide pura (real) ou ruído + senoide (fake)."""
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

    factory_mod = importlib.import_module(
        "app.domain.models.architectures.factory"
    )
    # Para Predictor, precisamos do AudioData e ProcessingStatus reais (do core).
    # Vamos carregar somente o módulo predictor + feature_preparer (não toda
    # cadeia DetectionService).
    # Como FeaturePreparer importa core.interfaces.audio, precisamos liberar
    # o stub fake do core e deixar import real:
    for k in list(sys.modules.keys()):
        if k.startswith("app.core") and isinstance(sys.modules[k], types.ModuleType) \
                and not hasattr(sys.modules[k], "__file__"):
            del sys.modules[k]

    # Idem para app.domain.services.detection: queremos os reais
    for k in list(sys.modules.keys()):
        if (
            k.startswith("app.domain.services.detection")
            or k == "app.domain.services"
            or k.startswith("app.domain.models")
        ) and isinstance(sys.modules[k], types.ModuleType) \
                and not hasattr(sys.modules[k], "__file__"):
            del sys.modules[k]

    audio_pre_mod = importlib.import_module(
        "app.domain.services.detection.audio_preprocessing"
    )

    # AudioData definido em core.interfaces.audio — pode haver dep sqlalchemy.
    # Tentamos import; se falhar, faz dataclass mínimo.
    try:
        from app.core.interfaces.audio import AudioData  # type: ignore
    except Exception:
        from dataclasses import dataclass

        @dataclass
        class AudioData:  # type: ignore
            samples: np.ndarray
            sample_rate: int
            duration: float
            channels: int = 1
            metadata: dict = None

    # ModelInfo: definido em detection.model_loader (tem dep TF). Importa.
    try:
        from app.domain.services.detection.model_loader import ModelInfo
    except Exception:
        # Stub mínimo
        class ModelInfo:  # type: ignore
            def __init__(self, **kw):
                for k, v in kw.items():
                    setattr(self, k, v)

    # FeaturePreparer + Predictor
    fp_mod = importlib.import_module(
        "app.domain.services.detection.feature_preparer"
    )
    pred_mod = importlib.import_module(
        "app.domain.services.detection.predictor"
    )

    # Cria modelo Keras
    spec = factory_mod.architecture_factory_registry.get_architecture_info(arch)
    input_type = spec.input_requirements.get("input_type", "spectrogram")
    if input_type == "raw_audio":
        input_shape = (SAMPLE_RATE * DURATION, 1)
    else:
        t_frames = int(np.ceil((SAMPLE_RATE * DURATION) / audio_pre_mod.DEFAULT_HOP))
        input_shape = (t_frames, audio_pre_mod.DEFAULT_N_MELS, 1)

    model = factory_mod.create_model_by_name(
        arch, input_shape=input_shape, num_classes=2,
    )

    # Constrói um ModelInfo mock (dataclass com campos do ModelInfo real)
    model_info = ModelInfo(
        name=f"test_{arch.lower().replace(' ', '_').replace('-', '_')}",
        architecture=arch,
        model=model,
        scaler=None,
        input_shape=input_shape,
        model_type='tensorflow',
        input_contract=None,
        temperature=1.0,
        jit_predict_fn=None,
        eer_threshold=None,
    )

    # FeaturePreparer dummy (não precisa de feature_service real para input_type-based)
    class _MockFeatureService:
        def extract_features(self, *a, **kw): return None
        def extract_segmented_features(self, *a, **kw): return None

    fp = fp_mod.FeaturePreparer(_MockFeatureService())
    predictor = pred_mod.Predictor()

    out = {}
    for kind in ["real", "fake"]:
        audio_samples = make_audio(kind)
        try:
            ad = AudioData(
                samples=audio_samples,
                sample_rate=SAMPLE_RATE,
                duration=float(DURATION),
                channels=1,
            )
        except Exception as e:
            tb = traceback.format_exc()
            return {
                "arch": arch, "ok": False,
                "error": f"AudioData ctor: {e}\n{tb[-500:]}",
            }
        try:
            prep = fp.prepare_input(ad, model_info, spec)
        except Exception as e:
            tb = traceback.format_exc()
            return {
                "arch": arch, "ok": False,
                "error": f"prep raised: {e}\n{tb[-500:]}",
            }
        if prep["status"] != "ok":
            return {
                "arch": arch, "ok": False,
                "error": f"prep failed for {kind}: {prep.get('error', '<no error msg>')}",
            }

        features = prep["features"]
        # Sanity-check do shape
        expected_dims = len(input_shape)
        if features.ndim != expected_dims:
            return {
                "arch": arch, "ok": False,
                "error": f"Shape mismatch: prep={features.shape} expected dims={expected_dims}",
            }

        try:
            pred_res = predictor.predict(model_info, features)
        except Exception as e:
            tb = traceback.format_exc()
            return {
                "arch": arch, "ok": False,
                "error": f"predict raised: {e}\n{tb[-500:]}",
            }
        if pred_res.status.value != "success":
            return {
                "arch": arch, "ok": False,
                "error": (
                    f"predict failed for {kind}: "
                    f"status={pred_res.status.value} errors={pred_res.errors}"
                ),
            }

        d = pred_res.data
        ok = (
            isinstance(d.get("is_deepfake"), bool)
            and "p_fake" in d
            and "p_real" in d
            and "confidence" in d
            and 0.0 <= d["p_fake"] <= 1.0 + 1e-3
            and 0.0 <= d["p_real"] <= 1.0 + 1e-3
            and abs(d["p_fake"] + d["p_real"] - 1.0) < 1e-2
        )
        out[kind] = {
            "ok": ok,
            "is_deepfake": d.get("is_deepfake"),
            "p_fake": d.get("p_fake"),
            "p_real": d.get("p_real"),
            "confidence": d.get("confidence"),
            "features_shape": tuple(features.shape),
        }

    all_ok = all(v["ok"] for v in out.values())
    return {"arch": arch, "ok": all_ok, "input_type": input_type, "results": out}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="", help="CSV (vazio=todos)")
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
                    print(
                        f"    [{kind:5s}] is_fake={v['is_deepfake']} "
                        f"p_fake={v['p_fake']:.4f} p_real={v['p_real']:.4f} "
                        f"conf={v['confidence']:.4f} feats={v['features_shape']}"
                    )
            else:
                print(f"  [FAIL] {r.get('error', 'unknown')}")
        except Exception as e:
            import traceback
            print(f"  [FAIL] {type(e).__name__}: {str(e)[:200]}")
            traceback.print_exc()
            results.append({"arch": arch, "ok": False, "error": str(e)})

    ok_count = sum(1 for r in results if r.get("ok"))
    print("\n" + "=" * 80)
    print(f"SUMÁRIO: {ok_count}/{len(results)} arquiteturas funcionais end-to-end")
    print("=" * 80)
    for r in results:
        st = "OK  " if r.get("ok") else "FAIL"
        print(f"  [{st}] {r.get('arch', '?')}")

    return 0 if ok_count == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
