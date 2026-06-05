"""Testes das otimizações Tier-1 (neutras em acurácia).

- #2 save sem otimizador: artefato menor, recarrega e prediz.
- #1 inferência ONNX: o Predictor usa a sessão ONNX quando presente e CAI para
  o TF em erro (provado com sessão mock — não exige onnxruntime instalado).
"""

from __future__ import annotations

import numpy as np
import pytest


# ───────────────────────── #2: save sem otimizador ─────────────────────────

def test_save_model_excludes_optimizer(tmp_path):
    import tensorflow as tf

    from app.core.config.settings import TrainingConfig
    from app.core.interfaces.base import ProcessingStatus
    from app.domain.models.training.trainer import ModelTrainer

    # Modelo com parâmetros suficientes p/ o estado do Adam pesar no arquivo.
    m = tf.keras.Sequential([
        tf.keras.layers.Input((16,)),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    m.compile(optimizer="adam", loss="binary_crossentropy")
    m.fit(np.random.rand(64, 16), np.random.randint(0, 2, 64), epochs=1, verbose=0)

    x = np.random.rand(8, 16).astype("float32")
    ref = m.predict(x, verbose=0)  # saída ANTES de salvar

    p_with = tmp_path / "with_opt.keras"
    m.save(str(p_with))  # baseline COM otimizador

    trainer = ModelTrainer(TrainingConfig())
    p_trainer = tmp_path / "trainer.keras"
    res = trainer.save_model(m, p_trainer)
    assert res.status == ProcessingStatus.SUCCESS
    assert p_trainer.exists()
    # Sem o estado do Adam (2 momentos/peso) → estritamente menor.
    assert p_trainer.stat().st_size < p_with.stat().st_size

    # Recarrega e prediz: saída IDÊNTICA (neutro em acurácia — só removeu o
    # estado do otimizador, pesos e grafo inalterados).
    loaded = tf.keras.models.load_model(str(p_trainer))
    out = loaded.predict(x, verbose=0)
    assert out.shape == ref.shape
    assert np.allclose(out, ref, atol=1e-5)
    # O otimizador do modelo original foi RESTAURADO após o save.
    assert m.optimizer is not None


# ───────────────────────── #1: caminho ONNX no Predictor ─────────────────────

def _tiny_model_info():
    import tensorflow as tf

    from app.domain.services.detection.model_loader import ModelInfo

    m = tf.keras.Sequential([
        tf.keras.layers.Input((4,)),
        tf.keras.layers.Dense(2, activation="softmax"),
    ])
    return ModelInfo(
        name="t", architecture="X", model=m, scaler=None,
        input_shape=(4,), model_type="tensorflow",
    )


class _FakeOnnx:
    def __init__(self, raise_it=False):
        self.raise_it = raise_it
        self.calls = 0

    def predict(self, x):
        self.calls += 1
        if self.raise_it:
            raise RuntimeError("onnx boom")
        # p(fake)=0.9 para todas as amostras (índice 1 = fake)
        return np.tile(np.array([[0.1, 0.9]], dtype="float32"), (len(x), 1))


def test_predictor_prefers_onnx_session_when_present():
    from app.core.interfaces.base import ProcessingStatus
    from app.domain.services.detection.predictor import Predictor

    mi = _tiny_model_info()
    mi.onnx_session = _FakeOnnx()
    r = Predictor().predict_batch(mi, [np.zeros(4, dtype="float32")])
    assert r.status == ProcessingStatus.SUCCESS
    assert mi.onnx_session.calls == 1  # usou o ONNX
    assert r.data[0]["p_fake"] > 0.8   # veio do ONNX (0.9), não do TF aleatório


def test_predictor_falls_back_to_tf_on_onnx_error():
    from app.core.interfaces.base import ProcessingStatus
    from app.domain.services.detection.predictor import Predictor

    mi = _tiny_model_info()
    mi.onnx_session = _FakeOnnx(raise_it=True)
    r = Predictor().predict_batch(mi, [np.zeros(4, dtype="float32")])
    # ONNX falhou → caiu para o TF, mas ainda retornou resultado válido
    assert r.status == ProcessingStatus.SUCCESS
    assert mi.onnx_session.calls == 1
    d = r.data[0]
    assert 0.0 <= d["p_fake"] <= 1.0 and "is_deepfake" in d


def test_predictor_no_onnx_uses_tf():
    from app.core.interfaces.base import ProcessingStatus
    from app.domain.services.detection.predictor import Predictor

    mi = _tiny_model_info()  # onnx_session None (default)
    r = Predictor().predict_batch(mi, [np.zeros(4, dtype="float32")])
    assert r.status == ProcessingStatus.SUCCESS
    assert 0.0 <= r.data[0]["p_fake"] <= 1.0
