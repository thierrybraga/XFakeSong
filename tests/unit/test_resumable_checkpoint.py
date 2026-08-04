"""Testes do checkpoint que sobrevive a retomadas de treino.

Contexto (2026-08-04): quedas de energia interromperam o benchmark durante o
RawNet2. O `BackupAndRestore` restaura pesos, otimizador e contador de épocas,
mas NÃO o estado dos demais callbacks: ao retomar, `ModelCheckpoint.best`
voltava a `None` e `MonitorCallback._is_improvement(x, None)` retorna `True`
incondicionalmente — a primeira época pós-retomada gravava por cima do melhor
checkpoint ainda que fosse pior. No run observado isso trocaria a época 30
(val_loss 0.113) pela 84 (val_loss 0.176), violando a declaração de protocolo
`checkpoint_selection: minimum_clean_validation_loss`.
"""

import json

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from app.domain.models.training.trainer import (  # noqa: E402
    EpochProgressLogger,
    ResumableModelCheckpoint,
)


def _tiny_model(seed: int = 0) -> "tf.keras.Model":
    tf.keras.utils.set_random_seed(seed)
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(4,)),
            tf.keras.layers.Dense(4, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    return model


def _data():
    rng = np.random.RandomState(0)
    x = rng.rand(16, 4).astype("float32")
    y = rng.rand(16, 1).astype("float32")
    return x, y


def _checkpoint(path):
    return ResumableModelCheckpoint(
        str(path),
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
        verbose=0,
    )


def test_persiste_o_melhor_valor_ao_lado_do_checkpoint(tmp_path):
    ckpt = tmp_path / "best.weights.h5"
    x, y = _data()
    callback = _checkpoint(ckpt)
    _tiny_model().fit(
        x, y, validation_data=(x, y), epochs=3, verbose=0, callbacks=[callback]
    )

    state = json.loads((tmp_path / "best.weights.h5.best.json").read_text())
    assert state["monitor"] == "val_loss"
    assert state["best"] == pytest.approx(callback.best)


def test_retomada_nao_sobrescreve_com_epoca_pior(tmp_path):
    """O cenário que motivou a correção: retomar e piorar."""
    ckpt = tmp_path / "best.weights.h5"
    x, y = _data()
    primeiro = _checkpoint(ckpt)
    _tiny_model().fit(
        x, y, validation_data=(x, y), epochs=3, verbose=0, callbacks=[primeiro]
    )
    assinatura = ckpt.read_bytes()

    # Processo novo (como após uma queda de energia), agora com val_loss pior.
    segundo = _checkpoint(ckpt)
    assert segundo.best is None, "callback recém-criado não conhece o histórico"
    _tiny_model(seed=7).fit(
        x,
        y,
        validation_data=(x, y * 100 + 50),
        epochs=2,
        verbose=0,
        callbacks=[segundo],
    )

    assert segundo.best == pytest.approx(primeiro.best)
    assert ckpt.read_bytes() == assinatura, "checkpoint bom foi sobrescrito"


def test_retomada_ainda_aceita_epoca_melhor(tmp_path):
    """A proteção não pode congelar o checkpoint: melhora tem de gravar."""
    ckpt = tmp_path / "best.weights.h5"
    x, y = _data()
    primeiro = _checkpoint(ckpt)
    _tiny_model().fit(
        x, y, validation_data=(x, y), epochs=2, verbose=0, callbacks=[primeiro]
    )
    assinatura = ckpt.read_bytes()

    segundo = _checkpoint(ckpt)
    segundo.set_model(_tiny_model(seed=3))
    segundo.on_train_begin()
    assert segundo.best == pytest.approx(primeiro.best)
    # Uma época comprovadamente melhor que o histórico deve passar.
    segundo.on_epoch_end(0, {"val_loss": primeiro.best / 2})

    assert segundo.best == pytest.approx(primeiro.best / 2)
    assert ckpt.read_bytes() != assinatura


def test_estado_ilegivel_nao_derruba_o_treino(tmp_path):
    """Uma queda no meio da gravação não pode impedir a retomada."""
    ckpt = tmp_path / "best.weights.h5"
    (tmp_path / "best.weights.h5.best.json").write_text("{lixo truncado")

    callback = _checkpoint(ckpt)
    callback.on_train_begin()

    assert callback.best is None


def test_estado_de_outra_metrica_e_ignorado(tmp_path):
    ckpt = tmp_path / "best.weights.h5"
    (tmp_path / "best.weights.h5.best.json").write_text(
        json.dumps({"monitor": "val_accuracy", "best": 0.9})
    )

    callback = _checkpoint(ckpt)
    callback.on_train_begin()

    assert callback.best is None, "val_accuracy não serve de baseline p/ val_loss"


def test_eta_usa_epocas_deste_processo_e_nao_o_indice_absoluto():
    """Após retomar na época 84, o ETA das 16 restantes não pode ser ~0."""
    import logging
    import re
    import time

    callback = EpochProgressLogger(label="RawNet2")
    callback.params = {"epochs": 100, "steps": 100}
    callback._started_at = time.time() - 15.7 * 60  # uma época desde o restart
    callback._epoch_started_at = time.time()

    capturado = []
    handler = logging.Handler()
    handler.emit = lambda record: capturado.append(record.getMessage())
    logger = logging.getLogger("training.progress")
    logger.addHandler(handler)
    try:
        callback.on_epoch_end(83, {"loss": 0.1})
    finally:
        logger.removeHandler(handler)

    eta = float(re.search(r"eta_min=([\d.]+)", capturado[-1]).group(1))
    assert eta == pytest.approx(15.7 * 16, rel=0.05)
