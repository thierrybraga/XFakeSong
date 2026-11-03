"""Fluxo funcional de detecção: do upload ao veredito.

SUJEITO: a jornada completa que um usuário percorre, atravessando interface,
serviço e modelo. Diferente dos testes de API (contrato de uma rota) e dos de
integração (dois módulos conversando): aqui o critério é o resultado que o
usuário vê.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.core.contracts.audio import AudioData
from app.core.contracts.base import ProcessingStatus
from app.domain.services.detection_service import DetectionService


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
    # Atributos que um ModelInfo REAL traz com defaults (Sprint 1.4/3.1/4.5 +
    # Tier-1 ONNX). Sem defini-los, o MagicMock devolve sub-mocks "truthy" e o
    # Predictor tenta usar onnx_session/temperature falsos (IndexError/TypeError).
    # Espelhar os defaults faz o teste exercitar o caminho TensorFlow real.
    fake_model_info.onnx_session = None
    fake_model_info.temperature = 1.0
    fake_model_info.eer_threshold = None
    fake_model_info.input_contract = None
    fake_model_info.jit_predict_fn = None

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


def test_limiar_calibrado_nao_contradiz_veredito_e_confianca(mock_detection_service):
    """P(fake) 0,8 com corte em 0,94: veredito real, com pontuação abaixo de 0,5."""
    info = mock_detection_service.model_loader.loaded_models["test_model"]
    info.eer_threshold = 0.94
    audio_data = AudioData(
        samples=np.random.rand(64600).astype(np.float32),
        sample_rate=16000,
        metadata={"source": "dummy.wav"},
        duration=4.0,
    )
    with patch.object(
        mock_detection_service.feature_preparer, "prepare_input"
    ) as mock_prep:
        mock_prep.return_value = {
            "status": "ok",
            "features": np.random.rand(1, 100, 80),
            "metadata": {},
        }
        result = mock_detection_service.detect_single(audio_data, "test_model")

    assert result.status == ProcessingStatus.SUCCESS
    assert result.data.is_fake is False, "a decisão continua sendo a do limiar"
    assert result.data.probabilities["fake"] == pytest.approx(0.8), "o bruto se mantém"
    pontuacao = 0.8 / 0.94 * 0.5
    assert result.data.metadata["fake_score"] == pytest.approx(pontuacao)
    assert result.data.confidence == pytest.approx(1 - pontuacao)


def test_analise_por_arquivo_mantem_limiar_e_ood(mock_detection_service):
    """Sem isto a aba Detectar anunciava "limiar fixo 0,5" para um corte de 0,94."""
    audio_data = AudioData(
        samples=np.random.rand(16000).astype(np.float32),
        sample_rate=16000,
        duration=1.0,
    )
    predicao = {
        "is_deepfake": False,
        "confidence": 0.2,
        "p_fake": 0.8,
        "p_real": 0.2,
        "classification_threshold": 0.94,
        "fake_score": 0.4255,
        "temperature_applied": 4.2,
        "is_ood": True,
    }
    with patch.object(
        mock_detection_service.feature_preparer,
        "prepare_input",
        return_value={"status": "ok", "features": np.zeros((1, 10)), "metadata": {"x": 1}},
    ), patch.object(
        mock_detection_service.predictor,
        "predict",
        return_value=MagicMock(status=ProcessingStatus.SUCCESS, data=predicao),
    ), patch.object(
        mock_detection_service.model_loader,
        "get_model",
        return_value=mock_detection_service.model_loader.loaded_models["test_model"],
    ):
        result = mock_detection_service._detect_with_config(audio_data, "test_model", [])

    assert result.status == ProcessingStatus.SUCCESS
    assert result.data.metadata["classification_threshold"] == 0.94
    assert result.data.metadata["fake_score"] == 0.4255
    assert result.data.metadata["is_ood"] is True
    assert result.data.metadata["x"] == 1, "os metadados do preparo continuam"
    assert result.data.probabilities["fake"] == 0.8


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
