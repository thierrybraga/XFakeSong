"""Rotas do front-end web respondem e servem a página.

SUJEITO: as rotas de template do FastAPI (`app/interfaces/web`), não a API
JSON. Guarda mínima contra um roteador quebrado passar despercebido.
"""

def test_api_bootstrap(client):
    """Bootstrap endpoint deve retornar status ok."""
    response = client.get('/api/v1/system/bootstrap')
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
