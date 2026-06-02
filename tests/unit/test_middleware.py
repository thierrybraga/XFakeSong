from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.middleware import setup_middleware


def test_system_middleware_adds_request_id_header(monkeypatch):
    monkeypatch.setenv("XFAKE_MAX_UPLOAD_MB", "1")
    app = FastAPI()
    setup_middleware(app)

    @app.get("/ping")
    async def ping():
        return {"ok": True}

    response = TestClient(app).get("/ping", headers={"X-Request-ID": "req-test"})

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "req-test"


def test_system_middleware_rejects_oversized_payload(monkeypatch):
    monkeypatch.setenv("XFAKE_MAX_UPLOAD_MB", "1")
    app = FastAPI()
    setup_middleware(app)

    @app.post("/upload")
    async def upload():
        return {"ok": True}

    response = TestClient(app).post(
        "/upload",
        content=b"x",
        headers={"Content-Length": str(2 * 1024 * 1024)},
    )

    assert response.status_code == 413
    assert response.headers["X-Request-ID"]
