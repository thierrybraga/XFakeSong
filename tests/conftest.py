import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock
import os
from app.main_fastapi import app
from app.dependencies import (
    get_detection_service,
    get_upload_service,
    get_training_service
)
from app.domain.services.detection_service import DetectionService
from app.domain.services.upload_service import AudioUploadService
from app.domain.services.training_service import TrainingService
from app.core.interfaces.base import ProcessingResult, ProcessingStatus
from app.schemas.api_models import DatasetMetadata

# Configurar API Key para testes
os.environ["XFAKESONG_API_KEY"] = "test-api-key"


@pytest.fixture
def api_key_headers():
    return {"X-API-Key": "test-api-key"}


@pytest.fixture
def mock_detection_service():
    mock = MagicMock(spec=DetectionService)
    mock.default_model = "test_model"
    mock.loaded_models = {"test_model": MagicMock()}
    mock.get_available_models.return_value = ["test_model"]
    mock.get_available_architectures.return_value = ["AASIST", "RawNet2"]
    return mock


@pytest.fixture
def mock_upload_service():
    mock = MagicMock(spec=AudioUploadService)
    mock.upload_directory = MagicMock()
    mock.upload_directory.exists.return_value = True

    # Mock create_dataset return
    def side_effect_create(name, type, desc):
        return DatasetMetadata(
            name=name,
            dataset_type=type,
            description=desc or "",
            file_count=0,
            total_size=0,
            total_duration=0.0,
            created_at=None,
            file_paths=[]
        )
    mock.create_dataset.side_effect = side_effect_create

    # Mock delete_dataset return
    mock.delete_dataset.return_value = ProcessingResult(
        status=ProcessingStatus.SUCCESS,
        metadata={"message": "Dataset deleted"}
    )

    return mock


@pytest.fixture
def mock_training_service():
    mock = MagicMock(spec=TrainingService)
    # Mock train_model return
    mock_result = MagicMock()
    mock_result.status = ProcessingStatus.SUCCESS
    mock_result.data = MagicMock()
    mock_result.data.metrics = {"accuracy": 0.95}
    mock_result.data.path = "/tmp/model.h5"
    mock.train_model.return_value = mock_result
    return mock


@pytest.fixture
def client(
    mock_detection_service,
    mock_upload_service,
    mock_training_service
):
    # Override dependencies
    app.dependency_overrides[get_detection_service] = \
        lambda: mock_detection_service
    app.dependency_overrides[get_upload_service] = \
        lambda: mock_upload_service
    app.dependency_overrides[get_training_service] = \
        lambda: mock_training_service

    with TestClient(app) as c:
        yield c

    # Clean up
    app.dependency_overrides = {}
