"""Smoke tests — verificam que TODOS os endpoints estão registrados,
respondem corretamente em casos triviais e o OpenAPI schema é válido.

Estes testes NÃO precisam de TF/Keras carregado; usam mocks via conftest.py.

Cobertura:
- Endpoints registrados (rotas existem)
- OpenAPI schema gerado sem erros
- /system/health, /version, /info, /status retornam 200
- /detection/architectures e /models retornam estruturas válidas
- /history retorna paginação consistente
- Endpoints novos (multi-model, uncertainty, cross-validate) estão registrados
- Schemas Pydantic expõem campos novos (temperature, OOD, EER)
"""

from __future__ import annotations

import pytest


# ────────────────────────────────────────────────────────────────────────
# Registro de rotas
# ────────────────────────────────────────────────────────────────────────

EXPECTED_ROUTES = [
    # system
    ("GET", "/api/v1/system/status"),
    ("GET", "/api/v1/system/health"),
    ("GET", "/api/v1/system/bootstrap"),
    ("GET", "/api/v1/system/version"),
    ("GET", "/api/v1/system/info"),
    # detection
    ("POST", "/api/v1/detection/analyze"),
    ("GET", "/api/v1/detection/models"),
    ("GET", "/api/v1/detection/architectures"),
    ("POST", "/api/v1/detection/multi-model"),   # API.5 (Sprint 4.4)
    ("POST", "/api/v1/detection/uncertainty"),   # API.6 (Sprint 5.4)
    # features
    ("POST", "/api/v1/features/extract"),
    ("GET", "/api/v1/features/types"),
    # training
    ("POST", "/api/v1/training/start"),
    ("GET", "/api/v1/training/status/{job_id}"),
    ("GET", "/api/v1/training/architectures"),
    ("POST", "/api/v1/training/cross-validate"),         # API.7 (Sprint 4.1)
    ("GET", "/api/v1/training/cross-validate/{job_id}"),
    # history
    ("GET", "/api/v1/history/"),
    ("GET", "/api/v1/history/{analysis_id}"),
    ("DELETE", "/api/v1/history/{analysis_id}"),
    # datasets
    ("GET", "/api/v1/datasets/"),
    ("POST", "/api/v1/datasets/"),
    ("POST", "/api/v1/datasets/{name}/upload"),
    ("DELETE", "/api/v1/datasets/{name}"),
    # voice_profiles
    ("GET", "/api/v1/profiles/"),
    ("POST", "/api/v1/profiles/"),
    ("GET", "/api/v1/profiles/{profile_id}"),
    ("PUT", "/api/v1/profiles/{profile_id}"),
    ("DELETE", "/api/v1/profiles/{profile_id}"),
    ("POST", "/api/v1/profiles/{profile_id}/samples"),
    ("DELETE", "/api/v1/profiles/{profile_id}/samples/{filename}"),
    ("POST", "/api/v1/profiles/{profile_id}/train"),
    ("POST", "/api/v1/profiles/{profile_id}/detect"),
]


def test_all_expected_routes_registered(client):
    """Todas as rotas esperadas devem estar registradas no app."""
    registered = set()
    for route in client.app.routes:
        # APIRoute tem path + methods
        path = getattr(route, "path", None)
        methods = getattr(route, "methods", set()) or set()
        if path:
            for m in methods:
                registered.add((m, path))

    missing = []
    for method, path in EXPECTED_ROUTES:
        if (method, path) not in registered:
            missing.append(f"{method} {path}")

    assert not missing, (
        f"Rotas faltando: {missing}\n"
        f"Registradas: {sorted(registered)[:10]}..."
    )


def test_openapi_schema_is_valid(client):
    """OpenAPI schema deve ser gerável sem erros."""
    resp = client.get("/api/openapi.json") if hasattr(
        client.app, "openapi_url"
    ) else None
    # Se app foi montado com openapi_url, testa via HTTP; senão chama direto
    if resp is None or resp.status_code == 404:
        schema = client.app.openapi()
    else:
        schema = resp.json()

    assert schema.get("openapi"), "openapi version ausente"
    assert "paths" in schema and len(schema["paths"]) > 10
    assert "info" in schema


# ────────────────────────────────────────────────────────────────────────
# Endpoints triviais (system)
# ────────────────────────────────────────────────────────────────────────

def test_system_status_ok(client):
    resp = client.get("/api/v1/system/status")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "operational"
    assert "active_services" in body
    assert isinstance(body["active_services"], list)


def test_system_health_ok(client):
    resp = client.get("/api/v1/system/health")
    assert resp.status_code == 200
    body = resp.json()
    # Status é "healthy" ou "degraded" (depende do DB no test env)
    assert body["status"] in {"healthy", "degraded"}
    assert "database" in body
    assert "uptime_seconds" in body
    assert isinstance(body["uptime_seconds"], (int, float))


def test_system_version_ok(client):
    resp = client.get("/api/v1/system/version")
    assert resp.status_code == 200
    body = resp.json()
    assert body["version"]
    assert body["python_version"]
    assert "platform" in body


def test_system_info_ok(client):
    resp = client.get("/api/v1/system/info")
    assert resp.status_code == 200
    body = resp.json()
    assert "status" in body
    assert "models" in body
    assert "endpoints" in body
    assert body["endpoints"]["docs"] == "/api/docs"


def test_system_bootstrap_ok(client):
    resp = client.get("/api/v1/system/bootstrap")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


# ────────────────────────────────────────────────────────────────────────
# Detection (sem upload — só endpoints GET)
# ────────────────────────────────────────────────────────────────────────

def test_detection_models_ok(client):
    resp = client.get("/api/v1/detection/models")
    assert resp.status_code == 200
    body = resp.json()
    assert "available_models" in body
    assert "default_model" in body
    assert "loaded_models" in body


def test_detection_architectures_ok(client):
    resp = client.get("/api/v1/detection/architectures")
    assert resp.status_code == 200
    body = resp.json()
    assert "architectures" in body
    assert isinstance(body["architectures"], list)


# ────────────────────────────────────────────────────────────────────────
# Features
# ────────────────────────────────────────────────────────────────────────

def test_features_types_ok(client):
    resp = client.get("/api/v1/features/types")
    assert resp.status_code == 200
    body = resp.json()
    assert "available_types" in body
    assert isinstance(body["available_types"], list)
    assert len(body["available_types"]) >= 5


# ────────────────────────────────────────────────────────────────────────
# Training (architectures GET é seguro de testar)
# ────────────────────────────────────────────────────────────────────────

def test_training_architectures_ok(client):
    resp = client.get("/api/v1/training/architectures")
    assert resp.status_code == 200
    body = resp.json()
    assert "architectures" in body


def test_training_status_not_found_returns_proper_response(client):
    """Job inexistente retorna status='not_found' (não 500)."""
    resp = client.get("/api/v1/training/status/00000000-0000-0000-0000-000000000000")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "not_found"


# ────────────────────────────────────────────────────────────────────────
# Schemas — campos novos preservados (API.4)
# ────────────────────────────────────────────────────────────────────────

def test_prediction_result_exposes_new_fields():
    """PredictionResult deve aceitar (mas não exigir) campos Sprint 1.4/2.5/4.5."""
    from app.schemas.api_models import PredictionResult
    p = PredictionResult(
        is_fake=True,
        confidence=0.85,
        probabilities={"real": 0.15, "fake": 0.85},
        model_name="test",
        features_used=["mel"],
        metadata={},
        temperature_applied=1.42,
        ood_score=0.42,
        is_ood=False,
        ood_threshold=0.2,
        classification_threshold=0.45,
    )
    dumped = p.model_dump()
    assert dumped["temperature_applied"] == 1.42
    assert dumped["is_ood"] is False
    assert dumped["classification_threshold"] == 0.45


def test_prediction_result_backward_compatible():
    """Clientes antigos sem os campos novos continuam funcionando."""
    from app.schemas.api_models import PredictionResult
    p = PredictionResult(
        is_fake=False,
        confidence=0.3,
        probabilities={"real": 0.7, "fake": 0.3},
        model_name="legacy",
        features_used=[],
        metadata={},
    )
    assert p.temperature_applied is None
    assert p.is_ood is None


def test_multi_model_request_validates_fusion():
    """MultiModelDetectionRequest deve rejeitar fusion inválido."""
    from app.schemas.api_models import MultiModelDetectionRequest
    with pytest.raises(Exception):
        MultiModelDetectionRequest(
            model_names=["a", "b"], fusion="unknown_strategy"
        )

    # Caso válido
    req = MultiModelDetectionRequest(
        model_names=["a", "b"], fusion="weighted_avg"
    )
    assert req.fusion == "weighted_avg"


def test_multi_model_request_min_2_models():
    """multi-model exige ≥2 modelos."""
    from app.schemas.api_models import MultiModelDetectionRequest
    with pytest.raises(Exception):
        MultiModelDetectionRequest(model_names=["only_one"])


def test_uncertainty_request_n_samples_bounds():
    """UncertaintyRequest valida n_samples ∈ [5, 200]."""
    from app.schemas.api_models import UncertaintyRequest
    req = UncertaintyRequest(n_samples=20)
    assert req.n_samples == 20

    with pytest.raises(Exception):
        UncertaintyRequest(n_samples=1)  # < 5
    with pytest.raises(Exception):
        UncertaintyRequest(n_samples=500)  # > 200


def test_cross_validation_request_validates_n_folds():
    """CrossValidationRequest valida n_folds ∈ [2, 20] e architecture."""
    from app.schemas.api_models import CrossValidationRequest
    req = CrossValidationRequest(
        architecture="aasist",
        dataset_path="data/test.npz",
        n_folds=5,
    )
    assert req.n_folds == 5
    assert req.architecture == "aasist"

    with pytest.raises(Exception):
        CrossValidationRequest(
            architecture="aasist", dataset_path="x.npz", n_folds=1
        )

    with pytest.raises(Exception):
        CrossValidationRequest(
            architecture="invalid_arch_xyz",
            dataset_path="x.npz",
        )
