#!/usr/bin/env python3
"""Verifica que os schemas Pydantic geram JSON Schema sem erros.

Testa PredictionResult e MultiModelPredictionResult — garante que
campos com tipos Union/Optional (ex: metadata) são serializáveis.

Uso:
    python scripts/tests/test_schema_fix.py
"""

import sys
from pathlib import Path

# Adiciona a raiz do projeto ao PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import os  # noqa: E402
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

print("=== Verificação de Schemas Pydantic ===")

from app.schemas.api_models import PredictionResult, MultiModelPredictionResult  # noqa: E402
from pydantic import TypeAdapter  # noqa: E402

ok = True

# Test 1: PredictionResult
try:
    schema = TypeAdapter(PredictionResult).json_schema()
    meta = schema.get("properties", {}).get("metadata", {})
    print(f"OK: PredictionResult  [metadata type={meta.get('type', 'any')}]")
except Exception as e:
    print(f"ERROR: PredictionResult — {e}")
    ok = False

# Test 2: MultiModelPredictionResult
try:
    schema2 = TypeAdapter(MultiModelPredictionResult).json_schema()
    meta2 = schema2.get("properties", {}).get("metadata", {})
    print(f"OK: MultiModelPredictionResult  [metadata type={meta2.get('type', 'any')}]")
except Exception as e:
    print(f"ERROR: MultiModelPredictionResult — {e}")
    ok = False

print("=== Concluído ===")
sys.exit(0 if ok else 1)
