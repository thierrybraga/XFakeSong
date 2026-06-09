"""Compatibilidade: treino do NOTEBOOK → pipeline de detecção do SISTEMA.

Os notebooks de treino (`pipeline/02`, `pipeline/03`) treinam via
`TrainingService.train_model(...)` a partir de um `.npz` e salvam o modelo + o
`input_contract`. Este teste garante que o artefato resultante é consumido pelo
caminho REAL de detecção do sistema — `DetectionService.detect_single(AudioData)`
sobre ÁUDIO BRUTO — e não só por `Predictor.predict` sobre um array já no shape
(o que `test_train_save_load_roundtrip` cobre).

Cobre as duas famílias treináveis pelo `TrainingService`:
- **espectrograma** (`MultiscaleCNN`) na forma (T, F) pequena que o `pipeline/02`
  usa, e
- **raw-audio** (`RawNet2`).

O `FeaturePreparer` do sistema precisa transformar o áudio bruto na forma de
entrada do modelo (via `input_contract` + registry) — é isso que validamos.

Pula sem TensorFlow (mesmo padrão dos outros testes de integração).
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("tensorflow")

from app.core.interfaces.audio import AudioData  # noqa: E402
from app.core.interfaces.base import ProcessingStatus  # noqa: E402

SR = 16000


@pytest.fixture(scope="module", autouse=True)
def _seed():
    import tensorflow as tf

    tf.keras.utils.set_random_seed(0)


def _synthetic_audio() -> AudioData:
    """1 s de áudio sintético (16 kHz) — entrada do pipeline real."""
    t = np.linspace(0, 1.0, SR, endpoint=False)
    y = (0.6 * np.sin(2 * np.pi * 220 * t)
         + 0.05 * np.random.default_rng(3).standard_normal(SR)).astype("float32")
    return AudioData(samples=y, sample_rate=SR, duration=1.0)


@pytest.mark.parametrize(
    "architecture, shape, name",
    [
        ("MultiscaleCNN", (32, 16), "nb_ms"),   # forma exata do pipeline/02
        ("RawNet2", (16000,), "nb_raw"),        # raw-audio
    ],
)
def test_notebook_trained_model_runs_in_detection_pipeline(
    tmp_path, architecture, shape, name
):
    from app.domain.services.training_service import TrainingService
    from app.domain.services.detection_service import DetectionService

    # 1. Treina como o notebook: TrainingService sobre um .npz sintético.
    rng = np.random.default_rng(0)
    n = 240
    X = rng.standard_normal((n, *shape)).astype("float32")
    y = rng.integers(0, 2, n).astype("int64")
    X[y == 1] += 0.6
    npz = tmp_path / "ds.npz"
    np.savez(npz, X_train=X, y_train=y)

    models_dir = tmp_path / "models"
    res = TrainingService(models_dir=str(models_dir)).train_model(
        architecture=architecture,
        dataset_path=str(npz),
        config={"epochs": 1, "batch_size": 32, "model_name": name},
    )
    assert res.status == ProcessingStatus.SUCCESS, res.errors

    # 2. Pipeline do sistema: detecção a partir de ÁUDIO BRUTO.
    ds = DetectionService(models_dir=str(models_dir), create_default_models=False)
    out = ds.detect_single(_synthetic_audio(), model_name=name)

    assert out.status == ProcessingStatus.SUCCESS, (
        f"{architecture}: detect_single falhou — {out.errors}"
    )
    result = out.data
    assert isinstance(result.is_fake, (bool, np.bool_))
    assert 0.0 <= float(result.confidence) <= 1.0, "confiança fora de [0,1]"
