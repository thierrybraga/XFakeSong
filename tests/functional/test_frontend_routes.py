"""Rotas de entrada encaminham à interface visual unificada."""

import pytest


def test_home_redirects_to_canonical_gradio_ui(client, monkeypatch):
    monkeypatch.delenv("XFAKE_API_ONLY", raising=False)
    monkeypatch.delenv("XFAKE_SKIP_GRADIO", raising=False)

    response = client.get("/", follow_redirects=False)

    assert response.status_code == 307
    assert response.headers["location"] == "/gradio/"


def test_legacy_loading_route_redirects_to_canonical_ui(client, monkeypatch):
    monkeypatch.delenv("XFAKE_API_ONLY", raising=False)
    monkeypatch.delenv("XFAKE_SKIP_GRADIO", raising=False)

    response = client.get("/loading", follow_redirects=False)

    assert response.status_code == 307
    assert response.headers["location"] == "/gradio/"


@pytest.mark.parametrize("flag", ["XFAKE_API_ONLY", "XFAKE_SKIP_GRADIO"])
def test_api_only_home_redirects_to_swagger_docs(client, monkeypatch, flag):
    monkeypatch.delenv("XFAKE_API_ONLY", raising=False)
    monkeypatch.delenv("XFAKE_SKIP_GRADIO", raising=False)
    monkeypatch.setenv(flag, "true")

    response = client.get("/", follow_redirects=False)

    assert response.status_code == 307
    assert response.headers["location"] == "/api/docs"
