"""Testes da restauração GUARDADA do melhor checkpoint (TrainingService).

Contexto (2026-07-14): o critério "melhor por val_loss" do ModelCheckpoint
pode selecionar uma época ruim e uma restauração corrompida pode produzir
NaN (observado no Res2Net: EER 14,9% → 50% após restaurar). A restauração
agora valida a val_loss do checkpoint contra a dos pesos em memória e
reverte quando o checkpoint degrada.
"""

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from app.domain.services.training_service import TrainingService  # noqa: E402


def _tiny_model(seed: int = 0) -> "tf.keras.Model":
    tf.keras.utils.set_random_seed(seed)
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(4,)),
            tf.keras.layers.Dense(8, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model


def _val_data(n: int = 64, seed: int = 1):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 4)).astype("float32")
    y = (X.sum(axis=1) > 0).astype("int32")
    return X, y


def test_restore_keeps_checkpoint_when_it_is_better(tmp_path):
    X, y = _val_data()
    model = _tiny_model()
    # Treina um pouco e salva o estado BOM como checkpoint.
    model.fit(X, y, epochs=8, verbose=0)
    good_weights = model.get_weights()
    ckpt = tmp_path / "best_checkpoint.keras"
    model.save(str(ckpt))

    # Corrompe os pesos em memória (estado "última época" ruim).
    model.set_weights([w + 5.0 for w in good_weights])

    restored = TrainingService._guarded_checkpoint_restore(
        model, ckpt, (X, y), batch_size=32
    )
    assert restored is True
    for got, expected in zip(model.get_weights(), good_weights):
        np.testing.assert_allclose(got, expected, rtol=1e-6)


def test_restore_reverts_when_checkpoint_is_worse(tmp_path):
    X, y = _val_data()
    model = _tiny_model()
    # Salva um checkpoint RUIM (pesos deslocados) e mantém os bons em memória.
    model.fit(X, y, epochs=8, verbose=0)
    good_weights = model.get_weights()

    model.set_weights([w + 5.0 for w in good_weights])
    ckpt = tmp_path / "best_checkpoint.keras"
    model.save(str(ckpt))
    model.set_weights(good_weights)

    restored = TrainingService._guarded_checkpoint_restore(
        model, ckpt, (X, y), batch_size=32
    )
    assert restored is False
    for got, expected in zip(model.get_weights(), good_weights):
        np.testing.assert_allclose(got, expected, rtol=1e-6)


def test_restore_reverts_when_checkpoint_produces_nan(tmp_path):
    X, y = _val_data()
    model = _tiny_model()
    model.fit(X, y, epochs=2, verbose=0)
    good_weights = model.get_weights()

    # Checkpoint com pesos NaN → predições/val_loss não-finitas.
    model.set_weights([np.full_like(w, np.nan) for w in good_weights])
    ckpt = tmp_path / "best_checkpoint.keras"
    model.save(str(ckpt))
    model.set_weights(good_weights)

    restored = TrainingService._guarded_checkpoint_restore(
        model, ckpt, (X, y), batch_size=32
    )
    assert restored is False
    for got in model.get_weights():
        assert np.isfinite(got).all()


def test_restore_supports_weights_only_checkpoint(tmp_path):
    """O benchmark agora grava best_checkpoint.weights.h5 (só pesos)."""
    X, y = _val_data()
    model = _tiny_model()
    model.fit(X, y, epochs=8, verbose=0)
    good_weights = model.get_weights()
    ckpt = tmp_path / "best_checkpoint.weights.h5"
    model.save_weights(str(ckpt))

    model.set_weights([w + 5.0 for w in good_weights])

    restored = TrainingService._guarded_checkpoint_restore(
        model, ckpt, (X, y), batch_size=32
    )
    assert restored is True
    for got, expected in zip(model.get_weights(), good_weights):
        np.testing.assert_allclose(got, expected, rtol=1e-6)


def test_restore_reverts_snapshot_on_load_failure(tmp_path):
    X, y = _val_data()
    model = _tiny_model()
    model.fit(X, y, epochs=2, verbose=0)
    good_weights = model.get_weights()

    # Arquivo inválido → load_weights levanta; snapshot deve ser restaurado.
    ckpt = tmp_path / "best_checkpoint.keras"
    ckpt.write_bytes(b"not a keras archive")

    restored = TrainingService._guarded_checkpoint_restore(
        model, ckpt, (X, y), batch_size=32
    )
    assert restored is False
    for got, expected in zip(model.get_weights(), good_weights):
        np.testing.assert_allclose(got, expected, rtol=1e-6)
