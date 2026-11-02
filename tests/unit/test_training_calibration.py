"""Calibração pós-treino e contrato de entrada, sem montar um `ModelTrainer`.

SUJEITO: `app.domain.models.training.calibration` e
`app.domain.models.training.artifacts.build_input_contract`, que saíram de
`trainer.py` (P2-15). Antes só eram exercitados através de um treino inteiro;
aqui cada regra é verificada isolada: o SNR realizado do AWGN, a recusa de
ruído fora da forma de onda e os campos que a inferência lê do contrato.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from app.domain.models.training import calibration  # noqa: E402
from app.domain.models.training.artifacts import build_input_contract  # noqa: E402


def _ondas(n=6, t=1600, semente=0):
    rng = np.random.default_rng(semente)
    return (0.3 * rng.standard_normal((n, t, 1))).astype("float32")


def test_awgn_realiza_exatamente_o_snr_pedido():
    X = _ondas()
    ruidoso = calibration.add_awgn(X, 10.0, seed=3)

    sinal = np.mean(X.reshape(len(X), -1) ** 2, axis=1)
    ruido = np.mean((ruidoso - X).reshape(len(X), -1) ** 2, axis=1)
    np.testing.assert_allclose(10 * np.log10(sinal / ruido), 10.0, atol=1e-3)
    np.testing.assert_array_equal(ruidoso, calibration.add_awgn(X, 10.0, seed=3))


@pytest.mark.parametrize(
    ("forma", "esperado"),
    [((4, 1600), True), ((4, 1600, 1), True), ((4, 100, 80), False), ((4, 63), False)],
)
def test_heuristica_de_forma_de_onda(forma, esperado):
    assert calibration.looks_like_waveform(np.zeros(forma)) is esperado


def test_calibracao_sob_ruido_so_na_forma_de_onda():
    config = SimpleNamespace(calibrate_under_noise=True, calibration_snr_db=[20, 10])
    y = np.array([0, 1, 0, 1, 0, 1])

    X_cal, y_cal = calibration.build_calibration_set((_ondas(), y), config)
    assert len(X_cal) == len(y_cal) == 3 * len(y)
    np.testing.assert_array_equal(y_cal, np.tile(y, 3))

    espectro = (np.zeros((6, 100, 80), "float32"), y)
    assert calibration.build_calibration_set(espectro, config) is espectro

    desligado = SimpleNamespace(calibrate_under_noise=False, calibration_snr_db=[20])
    val = (_ondas(), y)
    assert calibration.build_calibration_set(val, desligado) is val


def test_contrato_leva_a_calibracao_e_a_politica_da_arquitetura():
    modelo = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(1600, 1)),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(2, activation="softmax"),
        ]
    )

    contrato = build_input_contract(
        modelo,
        {"architecture": "AASIST"},
        scaler_applied=False,
        temperature=1.5,
        ood_threshold=-2.0,
        eer_threshold=0.42,
        eer_value=0.07,
    )
    assert (contrato["type"], contrato["format"]) == ("audio", "raw")
    assert contrato["input_shape"] == [1600, 1]
    assert contrato["temperature"] == 1.5 and contrato["scaler_applied"] is False
    assert (contrato["ood_threshold"], contrato["eer_threshold"]) == (-2.0, 0.42)
    assert contrato["eer_value"] == 0.07
    assert contrato["output_is_logits"] is False  # softmax na saída
    assert "crop_strategy" in contrato  # política do registry da arquitetura

    sem_calibracao = build_input_contract(modelo, scaler_applied=True)
    assert sem_calibracao["temperature"] == 1.0
    assert not {"ood_threshold", "eer_threshold", "eer_value"} & set(sem_calibracao)
