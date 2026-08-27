"""Fronteiras de confiança: autenticação e autorização da API.

SUJEITO: `app/core/auth` — que rota protegida sem chave recuse, que chave
inválida recuse, e que a validação aconteça ANTES de qualquer efeito
colateral. Least privilege verificado no ponto de entrada.
"""

import asyncio

import pytest
from fastapi import FastAPI, HTTPException

from app.core.auth.auth_handler import get_api_key, get_gradio_auth
from app.core.security import setup_security
from app.utils.file_utils import resolve_within_directory, validate_path_segment


@pytest.mark.parametrize(
    "value",
    [
        "",
        ".",
        "..",
        "../secret",
        r"..\secret",
        "%2e%2e%5csecret",
        "file:stream",
        "CON",
        "lpt1.txt",
        "name.",
        "/absolute",
        r"C:\secret",
        "name/child",
    ],
)
def test_validate_path_segment_rejects_traversal(value):
    with pytest.raises(ValueError):
        validate_path_segment(value)


def test_resolve_within_directory_keeps_path_inside_base(tmp_path):
    resolved = resolve_within_directory(tmp_path, "training", "dataset")

    assert resolved == (tmp_path / "training" / "dataset").resolve()
    assert tmp_path.resolve() in resolved.parents


def test_api_key_fails_closed_when_server_key_is_missing(monkeypatch):
    monkeypatch.delenv("XFAKESONG_API_KEY", raising=False)
    monkeypatch.delenv("XFAKE_ALLOW_INSECURE_DEV", raising=False)
    monkeypatch.setenv("DEEPFAKE_ENV", "development")

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(get_api_key(None))

    assert exc_info.value.status_code == 503


def test_api_key_allows_only_explicit_insecure_development(monkeypatch):
    monkeypatch.delenv("XFAKESONG_API_KEY", raising=False)
    monkeypatch.setenv("DEEPFAKE_ENV", "development")
    monkeypatch.setenv("XFAKE_ALLOW_INSECURE_DEV", "true")

    result = asyncio.run(get_api_key(None))

    assert result == "insecure-development"


def test_gradio_auth_is_required_in_production(monkeypatch):
    monkeypatch.setenv("DEEPFAKE_ENV", "production")
    monkeypatch.delenv("XFAKE_GRADIO_USERNAME", raising=False)
    monkeypatch.delenv("XFAKE_GRADIO_PASSWORD", raising=False)

    with pytest.raises(RuntimeError, match="obrigatória"):
        get_gradio_auth()


def test_gradio_auth_returns_configured_credentials(monkeypatch):
    monkeypatch.setenv("DEEPFAKE_ENV", "production")
    monkeypatch.setenv("XFAKE_GRADIO_USERNAME", "operator")
    monkeypatch.setenv("XFAKE_GRADIO_PASSWORD", "strong-password")

    assert get_gradio_auth() == ("operator", "strong-password")


def test_production_rejects_wildcard_cors(monkeypatch):
    monkeypatch.setenv("DEEPFAKE_ENV", "production")
    monkeypatch.setenv("ALLOWED_ORIGINS", "*")
    monkeypatch.setenv("ALLOWED_HOSTS", "example.test")

    with pytest.raises(RuntimeError, match="ALLOWED_ORIGINS"):
        setup_security(FastAPI())


def test_production_requires_allowed_hosts(monkeypatch):
    monkeypatch.setenv("DEEPFAKE_ENV", "production")
    monkeypatch.setenv("ALLOWED_ORIGINS", "https://example.test")
    monkeypatch.delenv("ALLOWED_HOSTS", raising=False)

    with pytest.raises(RuntimeError, match="ALLOWED_HOSTS"):
        setup_security(FastAPI())


def test_production_accepts_explicit_origins_and_hosts(monkeypatch):
    monkeypatch.setenv("DEEPFAKE_ENV", "production")
    monkeypatch.setenv("ALLOWED_ORIGINS", "https://example.test")
    monkeypatch.setenv("ALLOWED_HOSTS", "example.test")

    app = FastAPI()
    setup_security(app)

    assert len(app.user_middleware) >= 2
