"""Carregamento de modelo e predição no serviço de detecção.

SUJEITO: `app/domain/services/detection/` — resolver o artefato certo a partir
do contrato de inferência, carregá-lo e produzir uma pontuação. É onde um
`feature_frontend` incompatível entre artefato e preparo se manifesta.
"""

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


class _FixedTensorflowModel:
    def predict(self, X, verbose=0):
        del verbose
        values = np.asarray(X, dtype="float32").mean(axis=(1, 2))
        return np.stack([1.0 - values, values], axis=-1)


def test_tensorflow_predictor_averages_declared_multicrop():
    info = ModelInfo(
        name="bench_aasist",
        architecture="AASIST",
        model=_FixedTensorflowModel(),
        scaler=None,
        input_shape=(4, 1),
        model_type="tensorflow",
        input_contract={"crop_strategy": "train_random_eval_multicrop"},
        eer_threshold=0.5,
    )
    crops = np.stack([
        np.full((4, 1), 0.2, dtype="float32"),
        np.full((4, 1), 0.6, dtype="float32"),
        np.full((4, 1), 0.8, dtype="float32"),
    ])

    result = Predictor().predict(info, crops)

    assert result.status.value == "success"
    assert result.data["tta_crops"] == 3
    assert np.isclose(result.data["p_fake"], (0.2 + 0.6 + 0.8) / 3)
    assert result.data["is_deepfake"] is True


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
