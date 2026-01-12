import pytest
from app.main_startup import create_app


@pytest.fixture
def client():
    app = create_app()
    app.config['TESTING'] = True
    app.config['WTF_CSRF_ENABLED'] = False  # Disable CSRF for testing

    with app.test_client() as client:
        with app.app_context():
            yield client


def test_home_page(client):
    """Test response of home page."""
    response = client.get('/')
    # It might redirect to /gradio/ or /login or show index
    # Based on summary, it was verified to redirect to /gradio/ or show index.
    assert response.status_code in [200, 302]


def test_api_bootstrap(client):
    """Test API bootstrap endpoint."""
    response = client.get('/api/v1/system/bootstrap')
    assert response.status_code == 200
    assert response.json['status'] == 'ok'


# Auth routes are currently disabled in the application
# def test_login_page(client):
#     """Test response of login page."""
#     # Assuming /login exists based on auth controller
#     response = client.get('/auth/login')
#     # Or maybe just /login if blueprint prefix is empty?
#     # Usually auth blueprint has prefix /auth
#     if response.status_code == 404:
#         response = client.get('/login')
#
#     assert response.status_code in [200, 302]
