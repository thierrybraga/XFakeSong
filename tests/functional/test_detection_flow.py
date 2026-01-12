import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from app.domain.services.detection_service import DetectionService
from app.core.interfaces.base import ProcessingStatus
from app.core.interfaces.audio import AudioData


@pytest.fixture
def mock_detection_service(tmp_path):
    # Mock dependencies to avoid loading real weights but test the flow
    service = DetectionService(models_dir=str(tmp_path / "models"))

    # Mock ModelLoader to have a fake model
    mock_model = MagicMock()
    # Mock predict method of the keras model
    mock_model.predict.return_value = np.array([[0.8]])  # 80% fake

    fake_model_info = MagicMock()
    fake_model_info.model = mock_model
    # Needs to be a valid architecture name for registry lookup
    fake_model_info.architecture = "AASIST"
    fake_model_info.input_shape = (1, 64600)  # Example raw audio shape
    fake_model_info.model_type = "tensorflow"

    service.model_loader.loaded_models = {
        "test_model": fake_model_info
    }
    service.model_loader.default_model = "test_model"

    # Mock FeaturePreparer to avoid complex feature extraction if needed
    # But integration tests usually want to test that part too.
    # If we want to test FeaturePreparer, we need the feature service to work.

    return service


def test_detect_single_flow(mock_detection_service):
    # Create dummy AudioData
    audio_data = AudioData(
        samples=np.random.rand(64600).astype(np.float32),
        sample_rate=16000,
        metadata={"source": "dummy.wav"},
        duration=4.0
    )

    # We need to ensure FeaturePreparer can handle "AASIST" which usually
    # requires raw audio. The registry says AASIST requires 'spectrogram'.
    # If the registry says spectrogram, FeaturePreparer will try to extract.
    # Let's mock FeaturePreparer.prepare_input to skip actual extraction

    with patch.object(
        mock_detection_service.feature_preparer, 'prepare_input'
    ) as mock_prep:
        mock_prep.return_value = {
            'status': 'ok',
            'features': np.random.rand(1, 100, 80),  # Batch, Time, Feats
            'metadata': {}
        }

        result = mock_detection_service.detect_single(audio_data, "test_model")

        assert result.status == ProcessingStatus.SUCCESS
        assert result.data.is_fake is True  # 0.8 > 0.5
        assert result.data.confidence == 0.8
        assert result.data.model_name == "test_model"


def test_detect_model_not_found(mock_detection_service):
    audio_data = AudioData(
        samples=np.zeros(100),
        sample_rate=16000,
        metadata={"source": "dummy.wav"},
        duration=0.01
    )
    result = mock_detection_service.detect_single(
        audio_data, "non_existent_model"
    )
    assert result.status == ProcessingStatus.ERROR
