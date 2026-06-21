import numpy as np
import json

from app.domain.services.detection.model_loader import (
    ModelInfo,
    ModelLoader,
    TorchSSLOriginalModel,
)
from app.domain.services.detection.predictor import Predictor


class _FixedSklearnModel:
    def predict(self, X):
        return np.ones(len(X), dtype=int)

    def predict_proba(self, X):
        return np.tile(np.array([[0.52, 0.48]], dtype=float), (len(X), 1))


def test_sklearn_predictor_uses_model_eer_threshold():
    info = ModelInfo(
        name="bench_randomforest",
        architecture="RandomForest",
        model=_FixedSklearnModel(),
        scaler=None,
        input_shape=(37,),
        model_type="sklearn",
        input_contract={
            "type": "features",
            "format": "tabular",
            "eer_threshold": 0.45,
        },
        eer_threshold=0.45,
    )

    result = Predictor().predict(info, np.zeros(37, dtype="float32"))

    assert result.status.value == "success"
    assert result.data["classification_threshold"] == 0.45
    assert result.data["is_deepfake"] is True


def test_model_loader_infers_hybrid_and_randomforest_names(tmp_path):
    loader = ModelLoader(tmp_path, create_default_models=False)

    assert loader._infer_architecture_from_name("bench_randomforest") == "RandomForest"
    assert (
        loader._infer_architecture_from_name("bench_hybrid_cnn_transformer")
        == "Hybrid CNN-Transformer"
    )


def test_model_loader_registers_torch_ssl_original_pt_lazily(tmp_path):
    artifact = tmp_path / "bench_wavlm_original.pt"
    artifact.write_bytes(b"not-a-real-checkpoint")
    (tmp_path / "bench_wavlm_original_config.json").write_text(
        json.dumps(
            {
                "architecture": "WavLMOriginal",
                "input_shape": [16000, 1],
                "backbone_artifact": "wavlm_backbone",
            }
        ),
        encoding="utf-8",
    )

    loader = ModelLoader(tmp_path, create_default_models=False)
    loader._load_single_model(artifact, warmup=False)

    info = loader.loaded_models["bench_wavlm_original"]
    assert info.model_type == "pytorch_transformers"
    assert info.architecture == "WavLMOriginal"
    assert info.input_shape == (16000, 1)
    assert isinstance(info.model, TorchSSLOriginalModel)
    assert info.model._loaded is False


def test_load_available_models_discovers_without_loading_weights(tmp_path):
    artifact = tmp_path / "bench_svm.pkl"
    artifact.write_bytes(b"placeholder")
    (tmp_path / "bench_svm_config.json").write_text(
        json.dumps({"architecture": "SVM", "input_shape": [37]}),
        encoding="utf-8",
    )

    loader = ModelLoader(tmp_path, create_default_models=False)
    loader.load_available_models()

    assert loader.get_available_models() == ["bench_svm"]
    assert loader.default_model == "bench_svm"
    assert loader.loaded_models == {}
