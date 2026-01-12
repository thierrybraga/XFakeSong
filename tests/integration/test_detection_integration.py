import pytest
import numpy as np
import tensorflow as tf
import json
from app.domain.services.detection_service import DetectionService
from app.core.interfaces.audio import AudioData
from app.core.interfaces.base import ProcessingStatus


@pytest.fixture
def integration_models_dir(tmp_path):
    models_dir = tmp_path / "models"
    models_dir.mkdir()

    # Create a dummy model
    # Model that accepts spectrogram features (Time, Mel-bins)
    # AASIST usually takes (None, 80) or similar
    input_layer = tf.keras.layers.Input(shape=(None, 80))
    x = tf.keras.layers.GlobalAveragePooling1D()(input_layer)
    x = tf.keras.layers.Dense(10, activation='relu')(x)
    output_layer = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

    model_path = models_dir / "test_model_AASIST.h5"
    model.save(str(model_path))

    # Create config file
    config_path = models_dir / "test_model_AASIST_config.json"
    config = {
        "architecture": "AASIST",
        "input_shape": [None, 80],
        "model_type": "tensorflow"
    }
    with open(config_path, 'w') as f:
        json.dump(config, f)

    return models_dir


def test_detection_service_integration_flow(integration_models_dir):
    """
    Test full integration:
    1. Load model from disk (ModelLoader)
    2. Prepare features (FeaturePreparer - mocked or real if simple)
    3. Predict (Predictor)
    """

    # Initialize service with temp models dir
    service = DetectionService(models_dir=str(integration_models_dir))

    # Verify model loaded
    assert "test_model_AASIST" in service.loaded_models
    model_info = service.loaded_models["test_model_AASIST"]
    assert model_info.architecture == "AASIST"

    # Create dummy audio data
    audio_data = AudioData(
        samples=np.random.rand(64600).astype(np.float32),
        sample_rate=16000,
        metadata={"source": "dummy.wav"},
        duration=4.0
    )

    # For integration test, we might hit FeaturePreparer wanting to extract
    # features. If the architecture is AASIST, it typically uses raw audio.
    # We need to make sure FeaturePreparer handles this or we mock it slightly
    # to avoid needing complex dependencies like torchaudio if they are not
    # installed or configured.
    # However, let's try to run it 'as is' first to see where it breaks.
    # If FeaturePreparer fails, we'll know.

    result = service.detect_single(audio_data, "test_model_AASIST")

    # Check result
    if result.status != ProcessingStatus.SUCCESS:
        raise AssertionError(
            f"Integration test failed with errors: {result.errors}"
        )

    assert result.status == ProcessingStatus.SUCCESS
    assert result.data.model_name == "test_model_AASIST"
    assert "is_fake" in result.data.__dict__ or hasattr(result.data, "is_fake")
