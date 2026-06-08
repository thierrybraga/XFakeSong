"""Cobertura de `app/core/exceptions.py`.

Exercita (1) as classes de exceção de domínio e seus códigos/status, (2) o
montador RFC 7807 `_problem_detail` e (3) os 4 handlers globais via uma app
FastAPI mínima com TestClient.
"""

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

import app.core.exceptions as exc_mod
from app.core.exceptions import (
    AppError,
    AudioProcessingError,
    ConflictError,
    DatasetNotFoundError,
    FileTooLargeError,
    ModelNotFoundError,
    NotFoundError,
    ProfileNotFoundError,
    ServiceUnavailableError,
    TrainingError,
    UnsupportedFormatError,
    ValidationError,
    _problem_detail,
    setup_exception_handlers,
)


# ── classes de exceção ────────────────────────────────────────────────────

def test_app_error_defaults():
    e = AppError("boom")
    assert e.status_code == 500
    assert e.error_code == "INTERNAL_ERROR"
    assert e.extra == {}
    assert str(e) == "boom"


def test_not_found_with_and_without_identifier():
    with_id = NotFoundError("Modelo", "x")
    assert with_id.status_code == 404
    assert with_id.error_code == "NOT_FOUND"
    assert "Modelo" in with_id.detail and "x" in with_id.detail

    without = NotFoundError("Recurso")
    assert without.detail.endswith("não encontrado")


def test_validation_error_carries_field():
    e = ValidationError("inválido", field="name")
    assert e.status_code == 400
    assert e.error_code == "VALIDATION_ERROR"
    assert e.extra == {"field": "name"}


def test_validation_error_without_field():
    assert ValidationError("inválido").extra == {}


def test_conflict_service_audio_training_errors():
    assert ConflictError("dup").status_code == 409
    assert ServiceUnavailableError("db").status_code == 503
    assert "db" in ServiceUnavailableError("db").detail
    assert AudioProcessingError("x").error_code == "AUDIO_PROCESSING_ERROR"
    assert TrainingError("x").status_code == 400


def test_specialized_not_found_override_codes():
    assert ModelNotFoundError("m").error_code == "MODEL_NOT_FOUND"
    assert ModelNotFoundError("m").status_code == 404
    assert DatasetNotFoundError("d").error_code == "DATASET_NOT_FOUND"
    assert ProfileNotFoundError(7).error_code == "PROFILE_NOT_FOUND"
    assert "7" in ProfileNotFoundError(7).detail


def test_file_too_large_and_unsupported_format_messages():
    big = FileTooLargeError(50)
    assert big.status_code == 413
    assert "50" in big.detail

    fmt = UnsupportedFormatError("xyz", ["wav", "mp3"])
    assert fmt.status_code == 415
    assert "xyz" in fmt.detail
    assert "wav" in fmt.detail and "mp3" in fmt.detail


# ── _problem_detail (RFC 7807) ────────────────────────────────────────────

def test_problem_detail_minimal():
    body = _problem_detail(404, "Not Found", "missing")
    assert body == {
        "type": "about:blank",
        "title": "Not Found",
        "status": 404,
        "detail": "missing",
    }


def test_problem_detail_with_code_extra_and_request_id(monkeypatch):
    monkeypatch.setattr(exc_mod, "get_request_id", lambda: "req-123")
    body = _problem_detail(
        400, "T", "d", error_code="E", extra={"field": "f"}
    )
    assert body["error_code"] == "E"
    assert body["field"] == "f"
    assert body["request_id"] == "req-123"


# ── handlers globais via FastAPI ──────────────────────────────────────────

@pytest.fixture
def client():
    app = FastAPI()
    setup_exception_handlers(app)

    @app.get("/app-error")
    def _raise_app_error():
        raise ModelNotFoundError("xyz")

    @app.get("/http-error")
    def _raise_http():
        raise HTTPException(status_code=403, detail="forbidden")

    @app.get("/validation")
    def _needs_int(n: int):
        return {"n": n}

    @app.get("/boom")
    def _unhandled():
        raise RuntimeError("kaboom-internal")

    return TestClient(app, raise_server_exceptions=False)


def test_handle_app_error(client):
    r = client.get("/app-error")
    assert r.status_code == 404
    body = r.json()
    assert body["error_code"] == "MODEL_NOT_FOUND"
    assert body["status"] == 404


def test_handle_http_exception(client):
    r = client.get("/http-error")
    assert r.status_code == 403
    assert r.json()["detail"] == "forbidden"


def test_handle_validation_error(client):
    r = client.get("/validation")  # falta o parâmetro obrigatório 'n'
    assert r.status_code == 422
    body = r.json()
    assert body["error_code"] == "VALIDATION_ERROR"
    assert isinstance(body["errors"], list) and body["errors"]


def test_handle_unhandled_exception_hides_internal_detail(client):
    r = client.get("/boom")
    assert r.status_code == 500
    body = r.json()
    assert body["error_code"] == "INTERNAL_ERROR"
    # nunca vaza a mensagem interna da exceção
    assert "kaboom-internal" not in body["detail"]
