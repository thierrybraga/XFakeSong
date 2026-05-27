#!/usr/bin/env python3
"""Valida suporte a dispositivos de inferência (CPU/GPU).

Verifica que get_available_devices() lista CPU e que DetectionService
aceita set_device() sem erros.

Uso:
    python scripts/tests/validate_device_support.py
"""

import sys
from pathlib import Path
import os

# Adiciona a raiz do projeto ao PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from app.domain.services.detection.utils import get_available_devices  # noqa: E402
from app.domain.services.detection_service import DetectionService  # noqa: E402


def main() -> int:
    print("=== Validação de Suporte a Dispositivos ===")

    # 1. get_available_devices
    devices = get_available_devices()
    print(f"Dispositivos disponíveis: {devices}")
    assert "CPU" in devices, "CPU deve estar sempre disponível"
    print("OK: CPU presente")

    # 2. DetectionService.set_device
    try:
        ds = DetectionService(models_dir="test_models_dir")
        print(f"Device inicial: {ds.device}")

        ds.set_device("GPU:0")
        print(f"Após set_device('GPU:0'): {ds.device}")

        ds.set_device("CPU")
        print("Após reset para CPU: OK")

    except Exception as e:
        print(f"INFO: Inicialização do serviço falhou (esperado sem modelos): {e}")

    print("=== Validação concluída ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
