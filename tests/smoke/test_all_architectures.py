"""Smoke test — cria cada uma das 14 arquiteturas e faz 1 forward pass.

Objetivo: descobrir qual input_shape cada modelo realmente aceita e quais quebram
com a chamada padrão `create_model_by_name(arch, input_shape=(48000, 1), num_classes=2)`.

Uso standalone:
    python tests/smoke/test_all_architectures.py
    python tests/smoke/test_all_architectures.py --models AASIST,"Sonic Sleuth"

Uso via pytest (inclui mark smoke):
    pytest -m smoke tests/smoke/test_all_architectures.py
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np  # noqa: E402

# tests/smoke/ → tests/ → raiz
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(THIS_DIR))
sys.path.insert(0, PROJECT_ROOT)

# Carrega o módulo factory diretamente, evitando que app/domain/__init__
# dispare a cadeia de imports que requer sqlalchemy etc.
# Truque: criar packages "falsos" em sys.modules antes do import.
import importlib.util  # noqa: E402
import types  # noqa: E402

for _fake_pkg in [
    "app", "app.domain", "app.domain.models", "app.domain.models.architectures",
]:
    if _fake_pkg not in sys.modules:
        _m = types.ModuleType(_fake_pkg)
        _m.__path__ = [os.path.join(PROJECT_ROOT, *_fake_pkg.split("."))]
        sys.modules[_fake_pkg] = _m

SAMPLE_RATE = 16000
DURATION_SEC = 3
RAW_AUDIO_LEN = SAMPLE_RATE * DURATION_SEC  # 48000

SPEC_FRAMES = 128
SPEC_MELS = 80

CANDIDATE_INPUTS: dict[str, list[tuple]] = {
    "AASIST":                  [(RAW_AUDIO_LEN, 1), (RAW_AUDIO_LEN,)],
    "RawGAT-ST":               [(SPEC_FRAMES, SPEC_MELS, 1), (RAW_AUDIO_LEN, 1), (RAW_AUDIO_LEN,)],
    "RawNet2":                 [(RAW_AUDIO_LEN, 1), (RAW_AUDIO_LEN,)],
    "WavLM":                   [(RAW_AUDIO_LEN, 1), (RAW_AUDIO_LEN,)],
    "HuBERT":                  [(RAW_AUDIO_LEN, 1), (RAW_AUDIO_LEN,)],
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
    import importlib
    factory_mod = importlib.import_module("app.domain.models.architectures.factory")
    create_model_by_name = factory_mod.create_model_by_name

    try:
        model = create_model_by_name(arch, input_shape=input_shape, num_classes=2)
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

    return {"arch": arch, "ok": success is not None, "winning_shape": success, "attempts": attempts}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", default="", help="CSV de modelos a testar (vazio = todos)")
    args = parser.parse_args(argv)

    targets = [m.strip() for m in args.models.split(",")] if args.models else list(CANDIDATE_INPUTS.keys())

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

    print("\n" + "=" * 80)
    print("SUMÁRIO")
    print("=" * 80)
    ok_count = sum(1 for r in results if r["ok"])
    print(f"OK: {ok_count}/{len(results)}\n")
    for r in results:
        status = "OK  " if r["ok"] else "FAIL"
        shape = r["winning_shape"] or "—"
        print(f"  [{status}] {r['arch']:30s}  shape: {shape}")

    return 0 if ok_count == len(results) else 1


# ── pytest integration ────────────────────────────────────────────────────────
import pytest  # noqa: E402


@pytest.mark.smoke
def test_smoke_all_architectures() -> None:
    """Cria e faz forward pass em todas as 14 arquiteturas."""
    rc = main(argv=[])
    assert rc == 0, "Uma ou mais arquiteturas falharam — veja saída acima"


# ── standalone ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sys.exit(main())
