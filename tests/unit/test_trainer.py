import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from app.domain.models.training.trainer import ModelTrainer
from app.core.config.settings import TrainingConfig

from app.core.interfaces.base import ProcessingStatus


@pytest.fixture
def mock_trainer():
    config = TrainingConfig(
        batch_size=32,
        epochs=1,  # Fast for testing
        learning_rate=0.001
    )
    # Mock internal components to avoid complex initialization if needed
    # But initializing ModelTrainer seems safe as it just sets up config
    return ModelTrainer(config)


def test_trainer_initialization(mock_trainer):
    assert mock_trainer.config.epochs == 1
    assert mock_trainer.secure_pipeline is not None


def test_train_flow(mock_trainer):
    # Mock Keras Model
    mock_model = MagicMock()
    mock_model.fit.return_value = MagicMock(
        history={'loss': [0.5], 'accuracy': [0.8]}
    )
    # Mock predict return value for metric calculation (20 validation samples)
    mock_model.predict.return_value = np.zeros((20, 1))

    # Dummy Data
    X = np.random.rand(100, 10)
    y = np.random.randint(0, 2, 100)

    # We need to mock secure_pipeline.prepare_data to return split data
    with patch.object(
        mock_trainer.secure_pipeline, 'prepare_data'
    ) as mock_prep:
        # Create a mock that has .status == ProcessingStatus.SUCCESS
        mock_result = MagicMock()
        mock_result.status = ProcessingStatus.SUCCESS
        mock_result.data = {
            "X_train": X[:60], "y_train": y[:60],
            "X_val": X[60:80], "y_val": y[60:80],
            "X_test": X[80:], "y_test": y[80:]
        }
        mock_prep.return_value = mock_result

        result = mock_trainer.train(mock_model, (X, y))

        assert result.status == ProcessingStatus.SUCCESS
        mock_model.fit.assert_called_once()
