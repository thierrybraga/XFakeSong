from unittest.mock import MagicMock
from app.core.interfaces.base import ProcessingStatus


def test_list_models(client):
    response = client.get("/api/v1/detection/models")
    assert response.status_code == 200
    data = response.json()
    assert "test_model" in data["available_models"]
    assert data["default_model"] == "test_model"


def test_list_architectures(client):
    response = client.get("/api/v1/detection/architectures")
    assert response.status_code == 200
    data = response.json()
    assert "AASIST" in data["architectures"]


def test_analyze_audio_no_file(client):
    response = client.post("/api/v1/detection/analyze")
    assert response.status_code == 422  # Validation error


def test_analyze_audio_success(client, mock_detection_service):
    # Setup mock return
    mock_result = MagicMock()
    mock_result.status = ProcessingStatus.SUCCESS
    mock_result.data = MagicMock()
    mock_result.data.is_fake = True
    mock_result.data.confidence = 0.95
    mock_result.data.probabilities = {'fake': 0.95, 'real': 0.05}
    mock_result.data.model_name = "test_model"
    mock_result.data.features_used = ["raw"]
    mock_result.data.metadata = {}

    mock_detection_service.detect_from_file.return_value = mock_result

    # Create dummy file
    files = {'file': ('test.wav', b'fake audio data', 'audio/wav')}

    response = client.post("/api/v1/detection/analyze", files=files)

    assert response.status_code == 200
    data = response.json()
    assert data["is_fake"] is True
    assert data["confidence"] == 0.95
