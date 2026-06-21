import pytest
from unittest.mock import MagicMock, patch
import numpy as np
from app.domain.services.training_service import TrainingService
from app.core.interfaces.base import ProcessingStatus, ProcessingResult
from app.domain.models.architectures.registry import ArchitectureInfo


@pytest.fixture
def mock_dataset(tmp_path):
    # Create dummy data matching expected shape for tests
    # Assuming (samples, time, features) or (samples, features)
    X_train = np.random.rand(10, 20)
    y_train = np.random.randint(0, 2, 10)
    X_val = np.random.rand(5, 20)
    y_val = np.random.randint(0, 2, 5)

    file_path = tmp_path / "train_data.npz"
    np.savez(
        file_path,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val
    )
    return str(file_path)


@pytest.fixture
def training_service(tmp_path):
    return TrainingService(models_dir=str(tmp_path / "models"))


def test_training_flow_success(training_service, mock_dataset):
    # Mock architecture info
    with patch(
            "app.domain.services.training_service.get_architecture_info"
    ) as mock_get_info, \
            patch(
                "app.domain.services.training_service.importlib.import_module"
            ) as mock_import, \
            patch(
                "app.domain.services.training_service.ModelTrainer"
            ) as MockTrainer:

        mock_info = ArchitectureInfo(
            name="TEST_ARCH",
            description="Test Arch",
            supported_variants=["default"],
            input_requirements={},
            default_params={},
            module_path="app.architectures.test",
            function_name="create_model"
        )
        mock_get_info.return_value = mock_info

        # Mock model creation
        mock_module = MagicMock()
        mock_create_model = MagicMock()
        mock_module.create_model = mock_create_model
        mock_import.return_value = mock_module

        mock_model = MagicMock()
        mock_create_model.return_value = mock_model

        # Mock Trainer
        mock_trainer_instance = MockTrainer.return_value
        mock_trainer_instance.train.return_value = ProcessingResult(
            status=ProcessingStatus.SUCCESS,
            data={"accuracy": 0.9}
        )
        mock_trainer_instance.save_model.return_value = ProcessingResult(
            status=ProcessingStatus.SUCCESS,
            data="path/to/model"
        )

        config = {
            "model_name": "test_model_v1",
            "epochs": 1,
            "batch_size": 2,
            "parameters": {}
        }

        result = training_service.train_model(
            "TEST_ARCH", mock_dataset, config
        )

        assert result.status == ProcessingStatus.SUCCESS
        assert result.data is not None
        assert result.data.name == "test_model_v1"
        assert result.data.architecture == "TEST_ARCH"

        # Verify calls
        mock_create_model.assert_called_once()
        mock_trainer_instance.train.assert_called_once()
        mock_trainer_instance.save_model.assert_called_once()


def test_training_service_accepts_common_architecture_alias(tmp_path):
    service = TrainingService(models_dir=str(tmp_path / "models"))

    X_train = np.random.rand(10, 100, 80).astype("float32")
    y_train = np.array([0, 1] * 5)
    X_val = np.random.rand(4, 100, 80).astype("float32")
    y_val = np.array([0, 1, 0, 1])
    dataset = tmp_path / "spectrogram_transformer_alias.npz"
    np.savez(dataset, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)

    with patch(
        "app.domain.services.training_service.importlib.import_module"
    ) as mock_import, patch(
        "app.domain.services.training_service.ModelTrainer"
    ) as MockTrainer:
        mock_module = MagicMock()
        mock_model = MagicMock()
        mock_module.create_model.return_value = mock_model
        mock_import.return_value = mock_module

        trainer = MockTrainer.return_value
        trainer.train.return_value = ProcessingResult(
            status=ProcessingStatus.SUCCESS,
            data={"accuracy": 0.9},
            metadata={"model": mock_model},
        )
        trainer.save_model.return_value = ProcessingResult(
            status=ProcessingStatus.SUCCESS,
            data=str(tmp_path / "models" / "alias_model.keras"),
        )
        trainer._build_input_contract.return_value = {}

        result = service.train_model(
            "Spectrogram Transformer",
            str(dataset),
            {"model_name": "alias_model", "epochs": 1, "batch_size": 2},
        )

    assert result.status == ProcessingStatus.SUCCESS, result.errors
    assert result.data.architecture == "SpectrogramTransformer"
    mock_module.create_model.assert_called_once()


def test_training_flow_invalid_architecture(training_service, mock_dataset):
    with patch(
        "app.domain.services.training_service.get_architecture_info"
    ) as mock_get_info:
        mock_get_info.return_value = None

        result = training_service.train_model(
            "INVALID_ARCH", mock_dataset, {}
        )

        assert result.status == ProcessingStatus.ERROR
        assert "não encontrada" in result.errors[0]


def test_training_flow_dataset_not_found(training_service):
    with patch(
        "app.domain.services.training_service.get_architecture_info"
    ) as mock_get_info:
        mock_get_info.return_value = MagicMock()

        result = training_service.train_model(
            "TEST_ARCH", "non_existent.npz", {}
        )

        assert result.status == ProcessingStatus.ERROR
        assert "não encontrado" in result.errors[0]
