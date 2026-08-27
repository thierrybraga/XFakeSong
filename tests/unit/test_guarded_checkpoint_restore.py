"""Testes da restauração GUARDADA do melhor checkpoint (TrainingService).

Contexto (2026-07-14): o critério "melhor por val_loss" do ModelCheckpoint
pode selecionar uma época ruim e uma restauração corrompida pode produzir
NaN (observado no Res2Net: EER 14,9% → 50% após restaurar). A restauração
valida o checkpoint contra os pesos em memória e reverte quando degrada.

Contexto (2026-08-22): a validação usava val_loss SEMPRE, mesmo quando o
checkpoint tinha sido selecionado por `val_eer` — virando uma segunda
seleção, por critério diferente do declarado, e a segunda vencia. Na bateria
corrigida o Conformer perdeu por isso o checkpoint da época 50 (val_eer
1,374%, val_loss 0,2524) para os pesos da época 100 (val_eer 1,923%,
val_loss 0,1946). Os testes `val_eer` abaixo fixam o comportamento correto:
perda e EER medem coisas diferentes, então o guard compara pela métrica que
selecionou o checkpoint e mantém a proteção contra NaN em qualquer monitor.
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


# --------------------------------------------------------------------------
# monitor = val_eer
# --------------------------------------------------------------------------


def _imperfect_model(seed: int = 3):
    """Modelo que ERRA algumas amostras — necessário para separar loss de EER.

    Num modelo perfeito, afiar as probabilidades REDUZ a entropia cruzada; é
    justamente o erro que faz a confiança custar caro. Sem erro nenhum os dois
    critérios nunca discordariam e o teste não distinguiria as implementações.
    """
    X, y = _val_data(n=128, seed=seed)
    model = _tiny_model(seed=seed)
    model.fit(X, y, epochs=2, verbose=0)
    return model, X, y


def _sharpen_output_layer(model, fator: float = 8.0):
    """Escala kernel e bias da saída: MESMA ordenação, entropia muito pior.

    Multiplicar o logit por k > 0 é transformação monotônica — o EER não muda
    (o ranking é idêntico) — mas a perda dispara nas amostras erradas. É o
    descompasso calibração×ordenação em forma mínima.
    """
    pesos = model.get_weights()
    pesos[-2] = pesos[-2] * fator
    pesos[-1] = pesos[-1] * fator
    return pesos


def test_val_eer_monitor_keeps_checkpoint_que_val_loss_descartaria(tmp_path):
    """O caso Conformer: EER igual ou melhor, perda pior → o checkpoint FICA."""
    model, X, y = _imperfect_model()
    pesos_bons = model.get_weights()

    # Checkpoint = mesma ordenação, calibração pior (perda maior).
    model.set_weights(_sharpen_output_layer(model))
    ckpt = tmp_path / "best_checkpoint.weights.h5"
    model.save_weights(str(ckpt))
    pesos_ckpt = model.get_weights()

    perda_ckpt = model.evaluate(X, y, verbose=0, return_dict=True)["loss"]
    model.set_weights(pesos_bons)
    perda_memoria = model.evaluate(X, y, verbose=0, return_dict=True)["loss"]
    assert perda_ckpt > perda_memoria, (
        "cenário mal construído: o checkpoint precisa ter perda PIOR para o "
        "teste distinguir o guard por val_loss do guard por val_eer"
    )

    restored = TrainingService._guarded_checkpoint_restore(
        model, ckpt, (X, y), batch_size=32, monitor="val_eer"
    )
    assert restored is True
    for got, expected in zip(model.get_weights(), pesos_ckpt):
        np.testing.assert_allclose(got, expected, rtol=1e-6)


def test_val_loss_monitor_ainda_descarta_o_mesmo_checkpoint(tmp_path):
    """Contraprova: sob val_loss o comportamento antigo permanece."""
    model, X, y = _imperfect_model()
    pesos_bons = model.get_weights()

    model.set_weights(_sharpen_output_layer(model))
    ckpt = tmp_path / "best_checkpoint.weights.h5"
    model.save_weights(str(ckpt))
    model.set_weights(pesos_bons)

    restored = TrainingService._guarded_checkpoint_restore(
        model, ckpt, (X, y), batch_size=32, monitor="val_loss"
    )
    assert restored is False
    for got, expected in zip(model.get_weights(), pesos_bons):
        np.testing.assert_allclose(got, expected, rtol=1e-6)


def test_val_eer_monitor_reverte_quando_o_eer_piora(tmp_path):
    """A proteção continua existindo — só mudou a métrica que ela consulta."""
    model, X, y = _imperfect_model()
    pesos_bons = model.get_weights()

    # Inverte o sinal da saída: ordenação ao contrário, EER péssimo.
    pesos_ruins = list(pesos_bons)
    pesos_ruins[-2] = -pesos_ruins[-2]
    pesos_ruins[-1] = -pesos_ruins[-1]
    model.set_weights(pesos_ruins)
    ckpt = tmp_path / "best_checkpoint.weights.h5"
    model.save_weights(str(ckpt))
    model.set_weights(pesos_bons)

    restored = TrainingService._guarded_checkpoint_restore(
        model, ckpt, (X, y), batch_size=32, monitor="val_eer"
    )
    assert restored is False
    for got, expected in zip(model.get_weights(), pesos_bons):
        np.testing.assert_allclose(got, expected, rtol=1e-6)


def test_val_eer_monitor_reverte_checkpoint_nan(tmp_path):
    """NaN não produz EER não-finito, produz EER de acaso — a perda é que denuncia."""
    model, X, y = _imperfect_model()
    pesos_bons = model.get_weights()

    model.set_weights([np.full_like(w, np.nan) for w in pesos_bons])
    ckpt = tmp_path / "best_checkpoint.weights.h5"
    model.save_weights(str(ckpt))
    model.set_weights(pesos_bons)

    restored = TrainingService._guarded_checkpoint_restore(
        model, ckpt, (X, y), batch_size=32, monitor="val_eer"
    )
    assert restored is False
    for got in model.get_weights():
        assert np.isfinite(got).all()


def test_monitor_desconhecido_cai_para_val_loss_com_aviso(tmp_path, caplog):
    model, X, y = _imperfect_model()
    pesos_bons = model.get_weights()
    ckpt = tmp_path / "best_checkpoint.weights.h5"
    model.save_weights(str(ckpt))

    with caplog.at_level("WARNING"):
        restored = TrainingService._guarded_checkpoint_restore(
            model, ckpt, (X, y), batch_size=32, monitor="val_auc"
        )
    assert restored is True
    assert any("val_auc" in r.getMessage() for r in caplog.records), (
        "um monitor que o guard não sabe reavaliar precisa AVISAR antes de "
        "cair para val_loss — em silêncio ele reintroduz o descompasso"
    )
    for got, expected in zip(model.get_weights(), pesos_bons):
        np.testing.assert_allclose(got, expected, rtol=1e-6)
