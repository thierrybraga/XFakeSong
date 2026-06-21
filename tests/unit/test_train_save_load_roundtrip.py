"""Round-trip de treino → salvar → carregar → prever (TrainingService).

Trava os bugs encontrados na revisão do pipeline de treinamento:
- default_params do registry vazando hints de pipeline (patience…) p/ o
  construtor do modelo → crash.
- SecureTrainingPipeline aplicando StandardScaler em entradas 3D (espectrograma).
- prepare_data retornando ProcessingStatus.COMPLETED (inexistente).
- loss incompatível (binary_crossentropy × softmax de 2 unidades).
- métrica 'f1' não-nativa quebrando o compile.
- input_contract (temperature/EER/OOD) SOBRESCRITO no _config.json → calibração
  perdida no reload.

O teste treina um modelo minúsculo de verdade (1 época), então é mais lento que
um unit test puro, mas cobre o caminho real usado por API/Optuna/cross-val.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="module")
def trained_dir():
    """Treina um modelo via TrainingService e devolve (models_dir, metadata)."""
    from app.core.interfaces.base import ProcessingStatus
    from app.domain.services.training_service import TrainingService

    rng = np.random.default_rng(0)
    n, T, F = 400, 32, 16
    X = rng.standard_normal((n, T, F)).astype("float32")
    y = rng.integers(0, 2, n).astype("int64")
    X[y == 1] += 0.6  # separa as classes → treinável em 1 época

    tmp = tempfile.mkdtemp(prefix="rt_")
    tmp = Path(tmp)
    npz = tmp / "ds.npz"
    np.savez(npz, X_train=X, y_train=y)

    models_dir = tmp / "models"
    svc = TrainingService(models_dir=str(models_dir))
    res = svc.train_model(
        architecture="MultiscaleCNN",
        dataset_path=str(npz),
        config={"epochs": 1, "batch_size": 32, "model_name": "rt_probe"},
    )
    assert res.status == ProcessingStatus.SUCCESS, res.errors
    return models_dir, res.data


def test_artifacts_written(trained_dir):
    models_dir, _ = trained_dir
    assert (models_dir / "rt_probe.keras").exists()
    assert (models_dir / "rt_probe_config.json").exists()


def test_config_has_input_contract_with_calibration(trained_dir):
    """Bug A: o input_contract (com calibração) deve estar no config salvo."""
    models_dir, _ = trained_dir
    cfg = json.loads(
        (models_dir / "rt_probe_config.json").read_text(encoding="utf-8")
    )
    assert "input_contract" in cfg and isinstance(cfg["input_contract"], dict)
    ic = cfg["input_contract"]
    # Campos de calibração presentes (Sprints 1.4/2.5/4.5)
    assert "temperature" in ic
    assert "eer_threshold" in ic
    assert ic.get("architecture") == "MultiscaleCNN"


def test_loader_preserves_contract_on_reload(trained_dir):
    """Bug A: a calibração salva deve sobreviver ao reload (não resetar p/ 1.0)."""
    models_dir, _ = trained_dir
    cfg = json.loads(
        (models_dir / "rt_probe_config.json").read_text(encoding="utf-8")
    )
    saved_temp = float(cfg["input_contract"]["temperature"])
    saved_eer = float(cfg["input_contract"]["eer_threshold"])

    from app.domain.services.detection.model_loader import ModelLoader

    loader = ModelLoader(models_dir=str(models_dir))
    loader.load_available_models()
    mi = loader.get_model("rt_probe")
    assert mi is not None, "modelo não reconhecido pelo loader"
    assert mi.architecture == "MultiscaleCNN"
    assert mi.input_contract is not None, "input_contract perdido no reload"
    # Os valores recarregados batem EXATAMENTE com o salvo (não viraram default)
    assert mi.temperature == pytest.approx(saved_temp)
    assert mi.eer_threshold == pytest.approx(saved_eer)


def test_predictor_recognizes_and_predicts(trained_dir):
    """O Predictor reconhece o modelo recarregado e roda inferência."""
    models_dir, _ = trained_dir
    from app.domain.services.detection.model_loader import ModelLoader
    from app.domain.services.detection.predictor import Predictor

    loader = ModelLoader(models_dir=str(models_dir))
    loader.load_available_models()
    mi = loader.get_model("rt_probe")

    sample = np.random.default_rng(1).standard_normal((32, 16)).astype("float32")
    from app.core.interfaces.base import ProcessingStatus

    pr = Predictor().predict(mi, sample)
    assert pr.status == ProcessingStatus.SUCCESS, pr.errors
    d = pr.data
    # Campos essenciais do resultado + a calibração aplicada veio do contract
    for k in ("is_deepfake", "p_fake", "p_real", "confidence",
              "temperature_applied", "classification_threshold"):
        assert k in d
    assert 0.0 <= d["p_fake"] <= 1.0
    assert d["temperature_applied"] == pytest.approx(mi.temperature)


def test_aasist_save_load_roundtrip_uses_serializable_layers(tmp_path):
    """AASIST não deve depender de safe_mode=False nem de Lambda anônimo."""
    import tensorflow as tf

    from app.domain.models.architectures.aasist import create_model

    model = create_model(input_shape=(1024, 1), num_classes=2)
    path = tmp_path / "aasist_roundtrip.keras"
    model.save(path)

    loaded = tf.keras.models.load_model(path, compile=False)

    assert loaded.input_shape == (None, 1024, 1)
    assert loaded.output_shape == (None, 2)
    assert any(
        layer.name == "sinc_abs" and layer.__class__.__name__ == "MagnitudeLayer"
        for layer in loaded.layers
    )
    assert not any(layer.__class__.__name__ == "Lambda" for layer in loaded.layers)


def test_efficientnet_lstm_save_load_roundtrip_uses_serializable_layers(tmp_path):
    """EfficientNet-LSTM salvo em .keras deve carregar em safe mode."""
    import tensorflow as tf

    from app.domain.models.architectures.efficientnet_lstm import create_model

    model = create_model(
        input_shape=(100, 80),
        num_classes=2,
        lstm_units=16,
        dropout_rate=0.1,
        pretrained=False,
    )
    path = tmp_path / "efficientnet_lstm_roundtrip.keras"
    model.save(path)

    loaded = tf.keras.models.load_model(path, compile=False)

    assert loaded.input_shape == (None, 100, 80)
    assert loaded.output_shape == (None, 2)
    assert any(
        layer.__class__.__name__ == "TemporalPoolingLayer"
        for layer in loaded.layers
    )
    assert any(
        layer.__class__.__name__ == "DeltaFeatureLayer"
        for layer in loaded.layers
    )
