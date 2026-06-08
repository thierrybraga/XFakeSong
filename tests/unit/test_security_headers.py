"""Garante que os headers de segurança são aplicados a todas as respostas
e que HSTS/CSP permanecem opt-in (não quebram o Gradio por padrão).
"""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.middleware import _build_security_headers, setup_middleware


def _make_app() -> FastAPI:
    app = FastAPI()
    setup_middleware(app)

    @app.get("/ping")
    def _ping():
        return {"ok": True}

    return app


def test_security_headers_present_on_responses():
    client = TestClient(_make_app())
    r = client.get("/ping")
    assert r.status_code == 200
    assert r.headers["X-Content-Type-Options"] == "nosniff"
    assert r.headers["X-Frame-Options"] == "SAMEORIGIN"
    assert "Referrer-Policy" in r.headers
    assert "Permissions-Policy" in r.headers
    assert r.headers.get("X-Request-ID")  # request id continua presente


def test_defaults_have_no_hsts_or_csp(monkeypatch):
    monkeypatch.delenv("XFAKE_ENABLE_HSTS", raising=False)
    monkeypatch.delenv("XFAKE_CSP", raising=False)
    headers = _build_security_headers()
    assert "Strict-Transport-Security" not in headers
    assert "Content-Security-Policy" not in headers


def test_hsts_is_opt_in(monkeypatch):
    monkeypatch.setenv("XFAKE_ENABLE_HSTS", "1")
    assert "Strict-Transport-Security" in _build_security_headers()


def test_csp_is_opt_in(monkeypatch):
    monkeypatch.setenv("XFAKE_CSP", "default-src 'self'")
    assert _build_security_headers()["Content-Security-Policy"] == "default-src 'self'"
