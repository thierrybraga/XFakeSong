"""Regressões da revisão de treino (compile-respect, from_logits, augmentation).

Cobre os bugs P0/P1 encontrados na revisão das arquiteturas + processo de
treinamento:

P0.1/P0.2 — o `ModelTrainer` recompilava TODO modelo, descartando a loss e o
otimizador definidos pela arquitetura. Para o AASIST isso era fatal: a cabeça
AM-Softmax emite LOGITS CRUS e a string "sparse_categorical_crossentropy"
(from_logits=False) fazia o Keras clipar/normalizar logits como se fossem
probabilidades — treino matematicamente quebrado. Também matava o
WarmupCosineDecay dos Transformers e o LR baixo do fine-tune SSL.

P0.3 — o dataset aumentado (~2N amostras) era FINITO e o trainer fixava
`steps_per_epoch=N//B` sem `repeat()` → o iterator esgotava na ~2ª época e o
Keras interrompia o treino ("ran out of data").

P1.3 — as 7 técnicas de augmentation eram aplicadas a QUALQUER entrada
(SpecAugment em waveform, RawBoost em espectrograma — domínio errado).
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("tensorflow")

import tensorflow as tf  # noqa: E402

from app.core.config.settings import TrainingConfig  # noqa: E402
from app.domain.models.training.trainer import ModelTrainer  # noqa: E402


def _tiny_data(n=48, dim=8, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, dim)).astype("float32")
    y = rng.integers(0, 2, n).astype("int64")
    X[y == 1] += 1.0
    return X, y


def _tiny_model(units=1, activation="sigmoid"):
    return tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(8,)),
            tf.keras.layers.Dense(8, activation="relu"),
            tf.keras.layers.Dense(units, activation=activation),
        ]
    )


def _config(**over):
    base = dict(
        epochs=1,
        batch_size=16,
        use_augmentation=False,
        use_class_weighting=False,
    )
    base.update(over)
    return TrainingConfig(**base)


def test_trainer_preserves_precompiled_loss_and_optimizer():
    """P0.1/P0.2: modelo já compilado pela arquitetura NÃO é recompilado —
    loss (ex.: from_logits=True) e otimizador (ex.: AdamW) são preservados."""
    model = _tiny_model(units=2, activation=None)  # saída LINEAR (logits)
    arch_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    arch_opt = tf.keras.optimizers.AdamW(learning_rate=3e-4, weight_decay=0.01)
    model.compile(optimizer=arch_opt, loss=arch_loss, metrics=["accuracy"])

    X, y = _tiny_data()
    trainer = ModelTrainer(_config(), use_mixed_precision=False)
    res = trainer.train(model, (X, y), validation_data=(X[:16], y[:16]))

    assert res.status.value == "success", res.errors
    # Otimizador da arquitetura preservado (não substituído por Adam@1e-3)
    assert isinstance(model.optimizer, tf.keras.optimizers.AdamW), (
        f"otimizador da arquitetura foi descartado: {type(model.optimizer)}"
    )
    assert float(model.optimizer.learning_rate) == pytest.approx(3e-4), (
        "LR da arquitetura foi sobrescrito sem pedido explícito"
    )
    # Loss da arquitetura preservada (objeto from_logits, não a string do Keras)
    assert not isinstance(model.loss, str), (
        f"loss da arquitetura foi clobberada pela string: {model.loss!r}"
    )
    assert getattr(model.loss, "from_logits", None) is True


def test_trainer_applies_explicit_lr_on_precompiled_model():
    """P0.2: com lr_is_explicit (chamador pediu), o LR é aplicado SEM trocar
    o otimizador nem a loss da arquitetura."""
    model = _tiny_model(units=1, activation="sigmoid")
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    X, y = _tiny_data()
    trainer = ModelTrainer(
        _config(learning_rate=5e-4, lr_is_explicit=True),
        use_mixed_precision=False,
    )
    res = trainer.train(model, (X, y), validation_data=(X[:16], y[:16]))

    assert res.status.value == "success", res.errors
    assert isinstance(model.optimizer, tf.keras.optimizers.AdamW)
    assert float(model.optimizer.learning_rate) == pytest.approx(5e-4)


def test_resolve_loss_uses_from_logits_for_linear_output():
    """P0.1: modelo NÃO compilado com saída linear (logits crus, caso
    AM-Softmax) recebe loss objeto com from_logits=True — nunca a string."""
    model = _tiny_model(units=2, activation=None)  # linear → logits
    _, y = _tiny_data()
    trainer = ModelTrainer(_config(), use_mixed_precision=False)

    loss = trainer._resolve_loss(model, y)
    assert isinstance(loss, tf.keras.losses.SparseCategoricalCrossentropy)
    assert loss.from_logits is True

    # Sanidade: saída sigmoid continua recebendo a string compatível.
    sig = _tiny_model(units=1, activation="sigmoid")
    assert trainer._resolve_loss(sig, y) == "binary_crossentropy"


def test_training_with_augmentation_completes_all_epochs():
    """P0.3: com use_augmentation=True (default) e epochs>2, o treino deve
    completar TODAS as épocas — antes o iterator finito esgotava na ~2ª e o
    Keras interrompia ("ran out of data")."""
    model = _tiny_model(units=1, activation="sigmoid")
    model.compile(optimizer="adam", loss="binary_crossentropy",
                  metrics=["accuracy"])

    X, y = _tiny_data(n=64)
    trainer = ModelTrainer(
        _config(epochs=4, use_augmentation=True), use_mixed_precision=False
    )
    res = trainer.train(model, (X, y), validation_data=(X[:16], y[:16]))

    assert res.status.value == "success", res.errors
    history = (res.data or {}).get("history") or {}
    epochs_run = len(history.get("loss", []))
    assert epochs_run == 4, (
        f"treino truncado: {epochs_run}/4 épocas (iterator esgotado?)"
    )


@pytest.mark.parametrize(
    "shape, expects_rawboost, expects_specaug",
    [
        ((32, 16000), True, False),      # raw (N, T)
        ((32, 16000, 1), True, False),   # raw (N, T, 1)
        ((32, 100, 80), False, True),    # espectrograma (N, T, F)
    ],
)
def test_augmentation_selects_techniques_by_domain(
    shape, expects_rawboost, expects_specaug
):
    """P1.3: RawBoost/codec só em waveform; SpecAugment só em espectrograma."""
    from app.domain.models.training.augmentation import AudioAugmenter

    aug = AudioAugmenter({})
    names = {t.__name__ for t in aug._select_techniques(shape)}

    assert ("_rawboost" in names) is expects_rawboost
    assert ("_codec_simulation" in names) is expects_rawboost
    assert ("_frequency_mask" in names) is expects_specaug
    assert ("_time_mask" in names) is expects_specaug
    # Técnicas neutras valem para ambos os domínios.
    assert {"_add_noise", "_time_shift", "_volume_change"} <= names
