"""Testes unitários dos schemas Pydantic da API.

Verifica que os schemas geram JSON Schema válido sem erros de serialização
para tipos Union/Optional (campo metadata, etc.).
"""
import pytest
from pydantic import TypeAdapter

from app.schemas.api_models import MultiModelPredictionResult, PredictionResult


def test_prediction_result_schema() -> None:
    """PredictionResult deve gerar JSON Schema sem erro."""
    schema = TypeAdapter(PredictionResult).json_schema()
    assert "properties" in schema, "Schema deve ter 'properties'"
    assert "is_fake" in schema["properties"], "Campo 'is_fake' deve existir"
    assert "confidence" in schema["properties"], "Campo 'confidence' deve existir"


def test_prediction_result_metadata_field() -> None:
    """Campo metadata de PredictionResult deve ser serializável."""
    schema = TypeAdapter(PredictionResult).json_schema()
    props = schema.get("properties", {})
    assert "metadata" in props, "Campo 'metadata' deve existir no schema"


def test_multi_model_prediction_result_schema() -> None:
    """MultiModelPredictionResult deve gerar JSON Schema sem erro."""
    schema = TypeAdapter(MultiModelPredictionResult).json_schema()
    assert "properties" in schema, "Schema deve ter 'properties'"
    # A lista de resultados por modelo é exposta como `per_model` — é o campo
    # que o router (detection.py) preenche e que o serviço retorna em
    # metadata['per_model']. (Antes o teste checava 'results', que nunca existiu.)
    assert "per_model" in schema["properties"], "Campo 'per_model' deve existir"


def test_multi_model_metadata_field() -> None:
    """Campo metadata de MultiModelPredictionResult deve ser serializável."""
    schema = TypeAdapter(MultiModelPredictionResult).json_schema()
    props = schema.get("properties", {})
    assert "metadata" in props, "Campo 'metadata' deve existir no schema"
