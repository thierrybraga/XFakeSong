import pytest


def test_api_bootstrap(client):
    """Bootstrap endpoint deve retornar status ok."""
    response = client.get('/api/v1/system/bootstrap')
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
