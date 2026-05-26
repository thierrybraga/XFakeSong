#!/usr/bin/env python
"""Test the Pydantic schema fixes."""

print("=== Testing Pydantic Schema Fix ===")

from app.schemas.api_models import PredictionResult, MultiModelPredictionResult
from pydantic import TypeAdapter

# Test 1: PredictionResult schema
try:
    adapter = TypeAdapter(PredictionResult)
    schema = adapter.json_schema()
    print("OK: PredictionResult schema generated")
    if "metadata" in schema.get("properties", {}):
        meta_schema = schema["properties"]["metadata"]
        print(f"  metadata schema type: {meta_schema.get('type', 'unknown')}")
except Exception as e:
    print(f"ERROR: PredictionResult - {e}")

# Test 2: MultiModelPredictionResult schema
try:
    adapter2 = TypeAdapter(MultiModelPredictionResult)
    schema2 = adapter2.json_schema()
    print("OK: MultiModelPredictionResult schema generated")
    if "metadata" in schema2.get("properties", {}):
        meta_schema2 = schema2["properties"]["metadata"]
        print(f"  metadata schema type: {meta_schema2.get('type', 'unknown')}")
except Exception as e:
    print(f"ERROR: MultiModelPredictionResult - {e}")

print("=== Schema tests completed ===")
