"""Smoke test — cria cada uma das 14 arquiteturas e faz 1 forward pass.

Objetivo: descobrir qual input_shape cada modelo realmente aceita e quais quebram
com a chamada padrão `create_model_by_name(arch, input_shape=(48000, 1), num_classes=2)`.

Uso:
    python scripts/tests/test_all_architectures.py
    python scripts/tests/test_all_architectures.py --models AASIST,Sonic Sleuth
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback

# Evita warnings excessivos do TF
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np  # noqa: E402

# Path do projeto (scripts/tests/ → scripts/ → raiz)
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(THIS_DIR))
sys.path.insert(0, PROJECT_ROOT)

# Carrega o módulo factory diretamente, evitando que app/domain/__init__
# dispare a cadeia de imports que requer sqlalchemy etc.
# Truque: criar packages "falsos" em sys.modules antes do import.
import importlib.util  # noqa: E402
import types  # noqa: E402

for fake_pkg in [
    "app", "app.domain", "app.domain.models", "app.domain.models.architectures",
]:
    if fake_pkg not in sys.modules:
        m = types.ModuleType(fake_pkg)
        m.__path__ = [os.path.join(PROJECT_ROOT, *fake_pkg.split("."))]
        sys.modules[fake_pkg] = m

SAMPLE_RATE = 16000
DURATION_SEC = 3
RAW_AUDIO_LEN = SAMPLE_RATE * DURATION_SEC  # 48000

# Spectrograms tipicamente são (T_frames, n_mels) ou (T_frames, n_mels, 1)
# T_frames ≈ duration * SR / hop_length (com hop=512 e 3s @ 16kHz: ~94 frames)
SPEC_FRAMES = 128
SPEC_MELS = 80

# Tabela de inputs candidatos por modelo. Tentamos vários até um funcionar.
CANDIDATE_INPUTS = {
    # Modelos raw-audio (paper-faithful)
    "AASIST":                  [(RAW_AUDIO_LEN, 1), (RAW_AUDIO_LEN,)],
    "RawGAT-ST":               [(SPEC_FRAMES, SPEC_MELS, 1), (RAW_AUDIO_LEN, 1), (RAW_AUDIO_LEN,)],
    "RawNet2":                 [(RAW_AUDIO_LEN, 1), (RAW_AUDIO_LEN,)],
    "WavLM":                   [(RAW_AUDIO_LEN, 1), (RAW_AUDIO_LEN,)],
    "HuBERT":                  [(RAW_AUDIO_LEN, 1), (RAW_AUDIO_LEN,)],
    # Modelos espectrograma
    "Sonic Sleuth":            [(SPEC_FRAMES, SPEC_MELS, 1), (SPEC_FRAMES, SPEC_MELS)],
    "EfficientNet-LSTM":       [(SPEC_FRAMES, SPEC_MELS, 1), (SPEC_FRAMES, SPEC_MELS)],
    "MultiscaleCNN":           [(SPEC_FRAMES, SPEC_MELS, 1), (SPEC_FRAMES, SPEC_MELS)],
    "SpectrogramTransformer":  [(SPEC_FRAMES, SPEC_MELS, 1), (SPEC_FRAMES, SPEC_MELS)],
    "Conformer":               [(SPEC_FRAMES, SPEC_MELS), (SPEC_FRAMES, SPEC_MELS, 1)],
    "Hybrid CNN-Transformer":  [(SPEC_FRAMES, SPEC_MELS, 1), (SPEC_FRAMES, SPEC_MELS), (RAW_AUDIO_LEN, 1)],
    "Ensemble":                [(SPEC_FRAMES, SPEC_MELS, 1), (SPEC_FRAMES, SPEC_MELS)],
}


def _try_create(arch: str, input_shape: tuple) -> tuple[bool, str, object]:
    """Tenta criar um modelo. Returns (ok, message, model_or_None)."""
    # Import direto do módulo factory sem passar pelo app/__init__ chain
    # (que importa sqlalchemy em domain/models/analysis.py).
    import importlib
    factory_mod = importlib.import_module(
        "app.domain.models.architectures.factory"
    )
    create_model_by_name = factory_mod.create_model_by_name

    try:
        model = create_model_by_name(
            arch, input_shape=input_shape, num_classes=2,
        )
        # Forward pass com 1 sample
        import tensorflow as tf
        x = np.zeros((1, *input_shape), dtype=np.float32)
        y = model(x, training=False)
        out_shape = tuple(y.shape) if hasattr(y, "shape") else None
        return True, f"OK in={input_shape} out={out_shape} params={model.count_params():,}", model
    except Exception as e:
        return False, f"FAIL in={input_shape}: {type(e).__name__}: {str(e)[:200]}", None


def test_architecture(arch: str) -> dict:
    """Tenta cada candidato até um funcionar."""
    inputs = CANDIDATE_INPUTS.get(arch, [(RAW_AUDIO_LEN, 1)])
    attempts = []
    success = None

    for shape in inputs:
        ok, msg, _model = _try_create(arch, shape)
        attempts.append({"shape": shape, "ok": ok, "msg": msg})
        if ok:
            success = shape
            break

    return {
        "arch": arch,
        "ok": success is not None,
        "winning_shape": success,
        "attempts": attempts,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="", help="CSV de modelos a testar (vazio = todos)")
    args = parser.parse_args()

    if args.models:
        targets = [m.strip() for m in args.models.split(",")]
    else:
        targets = list(CANDIDATE_INPUTS.keys())

    print("=" * 80)
    print("Smoke test — 14 arquiteturas XFakeSong")
    print("=" * 80)

    results = []
    for arch in targets:
        print(f"\n>>> {arch}")
        result = test_architecture(arch)
        results.append(result)
        for a in result["attempts"]:
            status = "[OK]" if a["ok"] else "[FAIL]"
            print(f"  {status} {a['msg']}")

    # Sumário
    print("\n" + "=" * 80)
    print("SUMÁRIO")
    print("=" * 80)
    ok_count = sum(1 for r in results if r["ok"])
    print(f"OK: {ok_count}/{len(results)}")
    print()
    for r in results:
        status = "OK  " if r["ok"] else "FAIL"
        shape = r["winning_shape"] or "—"
        print(f"  [{status}] {r['arch']:30s}  shape: {shape}")

    return 0 if ok_count == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
