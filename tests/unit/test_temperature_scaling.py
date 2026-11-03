"""Temperatura sobre o logit real, com busca contínua.

SUJEITO: `app/domain/services/detection/temperature_scaling.py`. A grade antiga parava em
0,5 e 4,9, e o logit recuperado da probabilidade cortada em 1e-7 criava o teto de 0,97891
do AASIST (T=4,2).
"""

from __future__ import annotations

import numpy as np
import pytest

from app.core.contracts.base import ProcessingStatus
from app.domain.services.detection.temperature_scaling import (
    TEMPERATURE_MAX,
    TEMPERATURE_MIN,
    calibrated_probabilities,
    fit_temperature,
    logit_output_model,
    logits_from_probabilities,
    transported_threshold,
)


def _logits_com_temperatura(t_real: float, n: int = 6000, seed: int = 0):
    """Log-odds cuja probabilidade correta é sigmoid(z / t_real)."""
    rng = np.random.default_rng(seed)
    bem_calibrado = rng.normal(0.0, 3.0, n)
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-bem_calibrado))).astype(int)
    return bem_calibrado * t_real, y


@pytest.mark.parametrize("t_real", [0.2, 0.5, 1.0, 4.9, 12.0])
def test_recupera_a_temperatura_mesmo_fora_da_grade_antiga(t_real):
    z, y = _logits_com_temperatura(t_real)
    assert fit_temperature(np.stack([np.zeros_like(z), z], axis=1), y) == pytest.approx(
        t_real, rel=0.07
    )


def test_saida_sigmoide_e_softmax_dao_a_mesma_temperatura():
    z, y = _logits_com_temperatura(3.0)
    duas_classes = fit_temperature(np.stack([np.zeros_like(z), z], axis=1), y)
    assert fit_temperature(z[:, np.newaxis], y) == pytest.approx(duas_classes)
    assert fit_temperature(z, y) == pytest.approx(duas_classes)


def test_separacao_perfeita_nao_amplia_a_incerteza():
    """A perda é plana perto de zero: qualquer T baixa serve, desde que não suavize."""
    z = np.array([-50.0, 50.0] * 50)
    y = np.array([0, 1] * 50)
    assert TEMPERATURE_MIN <= fit_temperature(z, y) < 1.0


def test_busca_para_no_limite_e_avisa(caplog):
    z = np.array([-50.0, 50.0] * 50)
    y = np.array([1, 0] * 50)
    assert fit_temperature(z, y) == pytest.approx(TEMPERATURE_MAX, rel=0.02)
    assert "encostou no limite" in caplog.text


def test_teto_da_probabilidade_cortada_nao_existe_mais():
    """log-odds de 20 e 40 empatavam em 0,97891 com T=4,2."""
    logits = np.array([[0.0, 20.0], [0.0, 40.0]])
    p = calibrated_probabilities(logits, 4.2)[:, 1]
    assert p[0] > 0.99
    assert p[1] > p[0]

    cortado = 1.0 / (
        1.0 + np.exp(-logits_from_probabilities(np.array([1.0]))[0] / 4.2)
    )
    assert cortado == pytest.approx(0.97891, abs=1e-5)


def test_fora_da_saturacao_coincide_com_o_caminho_antigo():
    """Por isso o limiar já gravado continua válido com a mesma temperatura."""
    from app.domain.services.detection.predictor import apply_temperature_scaling

    rng = np.random.default_rng(1)
    z = rng.uniform(-7.0, 7.0, (500, 2))
    p = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
    np.testing.assert_allclose(
        calibrated_probabilities(z, 3.2),
        np.asarray(apply_temperature_scaling(p, 3.2)),
        atol=1e-9,
    )


def test_temperatura_invalida_nao_escala():
    z = np.array([[0.0, 2.0]])
    np.testing.assert_allclose(
        calibrated_probabilities(z, 0.0), calibrated_probabilities(z, 1.0)
    )


def test_saida_sigmoide_mantem_a_forma():
    assert calibrated_probabilities(np.array([[0.0], [2.0]]), 2.0).shape == (2, 1)


# ───────────────────────── modelo que devolve o logit ─────────────────────────


def _modelo(camada_final):
    import tensorflow as tf

    entrada = tf.keras.layers.Input((4,))
    oculta = tf.keras.layers.Dense(8, activation="tanh")(entrada)
    return tf.keras.Model(entrada, camada_final(oculta))


def _entrada():
    return np.random.default_rng(0).normal(size=(16, 4)).astype("float32")


def test_copia_sem_softmax_reproduz_o_modelo():
    import tensorflow as tf

    modelo = _modelo(tf.keras.layers.Dense(2, activation="softmax", dtype="float32"))
    logit_model = logit_output_model(modelo)
    assert logit_model is not modelo
    np.testing.assert_allclose(
        calibrated_probabilities(logit_model.predict(_entrada(), verbose=0)),
        modelo.predict(_entrada(), verbose=0),
        atol=1e-5,
    )


def test_copia_sem_sigmoide_reproduz_o_modelo():
    import tensorflow as tf

    modelo = _modelo(tf.keras.layers.Dense(1, activation="sigmoid"))
    np.testing.assert_allclose(
        calibrated_probabilities(
            logit_output_model(modelo).predict(_entrada(), verbose=0)
        ),
        modelo.predict(_entrada(), verbose=0),
        atol=1e-5,
    )


def test_camada_de_ativacao_separada():
    import tensorflow as tf

    entrada = tf.keras.layers.Input((4,))
    z = tf.keras.layers.Dense(2)(entrada)
    modelo = tf.keras.Model(entrada, tf.keras.layers.Activation("softmax")(z))
    np.testing.assert_allclose(
        calibrated_probabilities(
            logit_output_model(modelo).predict(_entrada(), verbose=0)
        ),
        modelo.predict(_entrada(), verbose=0),
        atol=1e-5,
    )


def test_modelo_que_ja_emite_logit_e_ele_mesmo():
    import tensorflow as tf

    modelo = _modelo(tf.keras.layers.Dense(2))
    assert logit_output_model(modelo) is modelo
    modelo = _modelo(tf.keras.layers.Activation("linear"))
    assert logit_output_model(modelo) is modelo


def test_sem_acesso_a_camada_final_devolve_none():
    import tensorflow as tf

    assert logit_output_model(object()) is None
    assert logit_output_model(_modelo(tf.keras.layers.Dense(2, activation="relu"))) is None


# ───────────────────────── caminho da inferência ─────────────────────────


def _model_info_saturado(temperature: float = 4.2):
    """Softmax com log-odds de 40: a saída do modelo já é exatamente 1 em float32."""
    import tensorflow as tf

    from app.domain.services.detection.model_loader import ModelInfo

    modelo = tf.keras.Sequential(
        [
            tf.keras.layers.Input((2,)),
            tf.keras.layers.Dense(
                2,
                activation="softmax",
                use_bias=False,
                kernel_initializer=tf.keras.initializers.Constant(
                    [[-10.0, 10.0], [10.0, -10.0]]
                ),
            ),
        ]
    )
    return ModelInfo(
        name="t",
        architecture="X",
        model=modelo,
        scaler=None,
        input_shape=(2,),
        model_type="tensorflow",
        temperature=temperature,
        eer_threshold=0.9,
    )


def test_inferencia_usa_o_logit_e_mantem_o_veredito():
    from app.domain.services.detection.predictor import Predictor

    entrada = [np.array([2.0, 0.0], dtype="float32")]
    info = _model_info_saturado()
    antes = Predictor().predict_batch(info, entrada)
    assert antes.status == ProcessingStatus.SUCCESS
    assert antes.data[0]["p_fake"] == pytest.approx(0.97891, abs=1e-5)

    info.logit_model = logit_output_model(info.model)
    depois = Predictor().predict_batch(info, entrada)
    assert depois.status == ProcessingStatus.SUCCESS
    assert depois.data[0]["p_fake"] > 0.999
    assert depois.data[0]["is_deepfake"] is antes.data[0]["is_deepfake"] is True


def test_sessao_onnx_continua_pela_probabilidade():
    from app.domain.services.detection.predictor import Predictor

    class _Onnx:
        def predict(self, x):
            return np.tile([0.1, 0.9], (len(x), 1)).astype("float32")

    info = _model_info_saturado(temperature=1.0)
    info.logit_model = logit_output_model(info.model)
    info.onnx_session = _Onnx()
    resultado = Predictor().predict_batch(info, [np.zeros(2, dtype="float32")])
    assert resultado.data[0]["p_fake"] == pytest.approx(0.9, abs=1e-6)


def test_scaler_calibra_sobre_o_logit_real():
    """Com a validação toda saturada, a probabilidade cortada não veria a escala certa."""
    from app.domain.services.detection.predictor import TemperatureScaler

    info = _model_info_saturado()
    rng = np.random.default_rng(0)
    # Log-odds de ±40 no modelo, com 10% de rótulos trocados: o melhor T é da ordem de
    # 40 / logit(0,9) = 18, longe de 16,1 / logit(0,9) = 7,3 da probabilidade cortada.
    x = np.tile([[2.0, 0.0], [-2.0, 0.0]], (200, 1)).astype("float32")
    y = np.tile([1, 0], 200)
    trocados = rng.choice(400, 40, replace=False)
    y[trocados] = 1 - y[trocados]

    scaler = TemperatureScaler()
    scaler.calibrate(info.model, x, y)
    assert scaler.temperature == pytest.approx(40.0 / np.log(9.0), rel=0.02)


def test_modelo_so_usa_logit_quando_o_contrato_declara(tmp_path):
    """Artefato anterior foi avaliado na escala cortada, e tem de reproduzir o seu run."""
    import json

    import tensorflow as tf

    from app.domain.services.detection.model_loader import ModelLoader

    modelo = _modelo(tf.keras.layers.Dense(2, activation="softmax"))
    for nome, declarado in (("antigo", False), ("novo", True)):
        modelo.save(tmp_path / f"{nome}.keras")
        contrato = {"temperature": 2.0, "eer_threshold": 0.5}
        if declarado:
            contrato["temperature_on_logits"] = True
        (tmp_path / f"{nome}_config.json").write_text(
            json.dumps(
                {
                    "architecture": "AASIST",
                    "input_shape": [4],
                    "input_contract": contrato,
                }
            ),
            encoding="utf-8",
        )
    carregador = ModelLoader(str(tmp_path))
    assert carregador.get_model("antigo").logit_model is None
    assert isinstance(carregador.get_model("novo").logit_model, tf.keras.Model)


# ───────────────────────── recalibração sem treinar ─────────────────────────


@pytest.mark.parametrize("nova", [0.4, 1.0, 6.6])
def test_limiar_transportado_mantem_o_veredito(nova):
    z = np.linspace(-15.0, 15.0, 601)
    limiar = 0.941
    antes = calibrated_probabilities(z[:, np.newaxis], 4.2).ravel() >= limiar
    depois = calibrated_probabilities(z[:, np.newaxis], nova).ravel() >= (
        transported_threshold(limiar, 4.2, nova)
    )
    np.testing.assert_array_equal(antes, depois)


def test_mesma_temperatura_mantem_o_limiar():
    assert transported_threshold(0.941, 4.2, 4.2) == pytest.approx(0.941)


def _logits_saturados(seed: int = 0, n: int = 600):
    """Maioria acima do corte de 16,1, com erros confiantes, como no AASIST."""
    rng = np.random.default_rng(seed)
    y = np.tile([0, 1], n // 2)
    s = (2 * y - 1) * np.abs(rng.normal(30.0, 12.0, n))
    trocados = rng.choice(n, n // 25, replace=False)
    s[trocados] = -s[trocados]
    return np.stack([np.zeros(n), s], axis=1), y


def test_recalibrar_nao_muda_veredito():
    """A perda não entra aqui: o corte também limitava o custo de um erro confiante, e
    a escala real pode perder para ele. Por isso o relatório traz antes e depois."""
    from tools.reporting.recalibrate_temperature import recalibrate

    logits, y = _logits_saturados()
    contrato = {"temperature": 4.2, "eer_threshold": 0.9, "ood_threshold": 0.02}
    r = recalibrate(logits, y, contrato)
    assert r["changed_verdicts"] == 0
    assert r["temperature"] > 4.9  # acima do teto da grade antiga
    assert r["saturated_fraction"] > 0.5
    assert set(r["nll"]) == {"before", "after"}
    assert r["previous"] == contrato


def test_erro_do_cartao_mede_confianca_contra_acerto():
    from tools.reporting.recalibrate_temperature import card_calibration_error

    y = np.array([0, 1] * 50)
    certo_e_seguro = np.where(y == 1, 1.0, 0.0)
    assert card_calibration_error(y, certo_e_seguro, 0.5) == pytest.approx(0.0)
    # Sempre certo, mas declarando confiança 0,75: erro de 0,25.
    hesitante = np.where(y == 1, 0.75, 0.25)
    assert card_calibration_error(y, hesitante, 0.5) == pytest.approx(0.25)


def test_gravar_preserva_o_contrato_e_guarda_o_anterior(tmp_path):
    import json

    from tools.reporting.recalibrate_temperature import recalibrate, write_sidecar

    logits, y = _logits_saturados()
    original = {
        "architecture": "AASIST",
        "input_contract": {
            "temperature": 4.2,
            "eer_threshold": 0.9,
            "ood_threshold": 0.02,
            "band_correction": {"cutoff_hz": 7500.0},
        },
    }
    sidecar = tmp_path / "bench_x_config.json"
    sidecar.write_text(json.dumps(original), encoding="utf-8")
    resultado = recalibrate(logits, y, original["input_contract"])
    write_sidecar(sidecar, resultado, tmp_path / "dados.npz")

    gravado = json.loads(sidecar.read_text(encoding="utf-8"))
    contrato = gravado["input_contract"]
    assert contrato["temperature"] == resultado["temperature"]
    assert contrato["eer_threshold"] == resultado["eer_threshold"]
    assert contrato["temperature_on_logits"] is True
    assert contrato["band_correction"] == {"cutoff_hz": 7500.0}
    registro = gravado["provenance"]["temperature_recalibration"]
    assert registro["previous"]["temperature"] == 4.2
    assert registro["partition"] == "validation"

    backup = tmp_path / "bench_x_config.json.pre-temperature.bak"
    assert json.loads(backup.read_text(encoding="utf-8")) == original
    # Uma segunda gravação não troca o original guardado.
    write_sidecar(sidecar, resultado, tmp_path / "dados.npz")
    assert json.loads(backup.read_text(encoding="utf-8")) == original


def test_validacao_nao_depende_do_teste(tmp_path):
    from benchmarks.validation_scores import load_validation

    caminho = tmp_path / "dados.npz"
    np.savez(
        caminho,
        y_train=np.zeros(3),
        X_val=np.zeros((2, 8, 1), dtype="float32"),
        y_val=np.array([0, 1]),
        # Treino, validação e o que viria depois, nessa ordem.
        speaker_ids=np.array(["T", "T", "T", "V1", "V2", "X"]),
    )
    validacao = load_validation(caminho)
    assert validacao.waveforms.shape == (2, 8)
    assert validacao.speakers.tolist() == ["V1", "V2"]
