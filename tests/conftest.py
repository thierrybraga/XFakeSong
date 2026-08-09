import os
import pathlib
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.core.contracts.base import ProcessingResult, ProcessingStatus
from app.dependencies import (
    get_detection_service,
    get_training_service,
    get_upload_service,
)
from app.domain.services.detection_service import DetectionService
from app.domain.services.training_service import TrainingService
from app.domain.services.upload_service import AudioUploadService
from app.interfaces.web.main_fastapi import app
from app.interfaces.web.schemas.api_models import DatasetMetadata

_TEST_CATEGORIES = ("unit", "api", "functional", "integration", "smoke")


def pytest_collection_modifyitems(config, items):
    """Marca cada teste pela categoria da pasta onde ele vive."""
    for item in items:
        parts = set(pathlib.Path(str(item.fspath)).parts)
        for cat in _TEST_CATEGORIES:
            if cat in parts:
                item.add_marker(getattr(pytest.mark, cat))
                break

# Configurar API Key para testes
os.environ["XFAKESONG_API_KEY"] = "test-api-key"


@pytest.fixture(autouse=True)
def _reset_tf_mixed_precision_policy():
    """BUG FIX (flakiness sistêmica por estado global do Keras): a suíte

    completa expôs testes que passam isolados mas falham quando rodados
    depois de outro teste que chama
    ``tf.keras.mixed_precision.set_global_policy("mixed_float16")`` sem
    reverter — a política é global de PROCESSO, não de teste, então vaza
    para qualquer teste seguinte na mesma sessão pytest.

    Dois sintomas já confirmados (causa raiz idêntica, reproduzida
    isoladamente): ``test_trainer_preserves_precompiled_loss_and_optimizer``
    (o Keras encapsula o otimizador precompilado num ``LossScaleOptimizer``
    ao chamar ``model.compile()`` sob a política vazada) e
    ``test_architecture_builds_and_forwards`` para AASIST/RawGAT-ST/Ensemble
    (camadas customizadas — ``SincConvLayer`` e afins — produzem saída
    não-finita sob ``float16``). Reseta para ``float32`` antes de cada teste
    e restaura a política anterior depois, para não mascarar os testes que
    *de fato* querem exercitar mixed precision (ex.:
    ``test_sinc_layers_mixed_precision.py``, que já gerencia a própria
    política com ``try/finally``).
    """
    try:
        import tensorflow as tf
    except ImportError:
        yield
        return
    original = tf.keras.mixed_precision.global_policy()
    tf.keras.mixed_precision.set_global_policy("float32")
    yield
    tf.keras.mixed_precision.set_global_policy(original)


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
def mock_upload_service(tmp_path):
    mock = MagicMock(spec=AudioUploadService)
    mock.upload_directory = tmp_path / "uploads"
    mock.upload_directory.mkdir()
    mock.SUPPORTED_FORMATS = AudioUploadService.SUPPORTED_FORMATS
    mock.MAX_FILE_SIZE = AudioUploadService.MAX_FILE_SIZE

    # Mock create_dataset return.
    # O serviço REAL retorna ProcessingResult[DatasetMetadata] (não o
    # DatasetMetadata cru), e o router faz result.status / result.data.
    # O mock precisa espelhar esse contrato senão result.status estoura
    # AttributeError.
    def side_effect_create(name, type, desc):
        return ProcessingResult(
            status=ProcessingStatus.SUCCESS,
            data=DatasetMetadata(
                name=name,
                dataset_type=type,
                description=desc or "",
                file_count=0,
                total_size=0,
                total_duration=0.0,
                created_at=None,
                file_paths=[]
            ),
            metadata={"message": f"Dataset {name} criado com sucesso"},
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
