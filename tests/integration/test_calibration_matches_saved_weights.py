"""A calibração gravada tem de pertencer aos pesos que foram SALVOS.

Por que este teste existe (2026-07-28): o `TrainingService` calibra
temperatura e limiar de EER dentro de `trainer.train()` — ou seja, sobre os
pesos da ÚLTIMA época — e só depois restaura o melhor checkpoint (por
`val_loss`), que é o que vai para o disco e para a produção.

A coerência é garantida por um bloco de RECALIBRAÇÃO em
`training_service.train_model`, executado após a restauração bem-sucedida.
Inverter essa ordem, ou remover esse bloco, faria o `eer_threshold` do
contrato pertencer a um modelo diferente do entregue — sem erro, sem aviso, e
visível só depois de um treino completo. O `Predictor` decide com esse limiar,
então o efeito chega direto à interface.

O teste treina de verdade, força a restauração e recalcula o limiar a partir
do artefato em disco. Se a ordem se inverter, ele falha em segundos em vez de
depois de dias de GPU.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("tensorflow")


def _dataset(tmp: Path) -> Path:
    """Pequeno e SEPARÁVEL SÓ EM PARTE — de propósito.

    Com separação forte o EER vai a zero, e aí o limiar não é único: qualquer
    ponto do intervalo entre as classes serve, e diferenças de 1e-7 (batch de
    predição diferente, por exemplo) escolhem outro ponto. O teste passaria ou
    falharia por acaso. Com sobreposição, o limiar de EER é único e a
    comparação tem significado.
    """
    rng = np.random.default_rng(0)
    n = 240
    y = np.array([0, 1] * (n // 2))
    X = rng.normal(0, 1, (n, 100, 80, 1)).astype("float32")
    X[y == 1] += 0.05
    npz = tmp / "ds.npz"
    np.savez(
        npz,
        X_train=X[:160], y_train=y[:160],
        X_val=X[160:], y_val=y[160:],
    )
    return npz


def test_contract_calibration_belongs_to_the_saved_model():
    import tensorflow as tf

    from app.domain.models.training.metrics import MetricsCalculator
    from app.domain.services.detection.predictor import (
        apply_temperature_scaling,
        normalize_logits_to_probs,
    )
    from app.domain.services.training_service import TrainingService

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        npz = _dataset(tmp)
        service = TrainingService(models_dir=str(tmp))
        result = service.train_model(
            architecture="Sonic Sleuth",
            dataset_path=str(npz),
            config={
                "epochs": 6,
                "batch_size": 16,
                "model_name": "t",
                "verbose": 0,
                # força o caminho do melhor checkpoint (o do benchmark)
                "checkpoint_path": str(tmp / "best.weights.h5"),
                "early_stopping": False,
            },
        )
        assert result.status.name == "SUCCESS", result.errors

        contract = json.loads(
            (tmp / "t_config.json").read_text(encoding="utf-8")
        )["input_contract"]
        assert "eer_threshold" in contract, "contrato sem limiar operacional"
        assert contract.get("eer_value", 0.0) > 1e-6, (
            "EER degenerado (=0): o limiar nao e unico e a comparacao abaixo "
            "nao teria significado — ajuste a separacao do fixture"
        )
        temperature = float(contract.get("temperature", 1.0))

        # Recalcula o limiar A PARTIR DO ARTEFATO, no mesmo conjunto de val.
        # `np.load` mantem o handle do npz aberto (leitura preguicosa dos
        # membros) e no Windows isso impede o TemporaryDirectory de limpar —
        # copia para memoria e fecha.
        with np.load(npz) as arquivo:
            X_val = np.array(arquivo["X_val"])
            y_val = np.array(arquivo["y_val"])
        model = tf.keras.models.load_model(tmp / "t.keras")
        probs = normalize_logits_to_probs(
            # mesmo batch do trainer: com scores saturados, batches diferentes
            # mudam o resultado no ultimo digito e desempatam em outro ponto
            model.predict(X_val, batch_size=16, verbose=0),
            contract.get("output_is_logits"),
        )
        probs = np.asarray(
            apply_temperature_scaling(probs, temperature), dtype="float64"
        )
        scores = (
            probs[:, 1] if probs.ndim > 1 and probs.shape[-1] > 1
            else probs.ravel()
        )
        _eer, threshold = MetricsCalculator().calculate_eer(y_val, scores)

        assert threshold == pytest.approx(
            float(contract["eer_threshold"]), abs=1e-4
        ), (
            "o limiar do contrato nao corresponde aos pesos salvos: a "
            "calibracao voltou a ser feita ANTES da restauracao do melhor "
            "checkpoint (ver training_service.train_model)"
        )


def test_contract_declares_whether_the_output_is_logits():
    """Treino e inferência não podem decidir isso por critérios diferentes."""
    from app.domain.models.architectures.factory import create_model_by_name
    from app.domain.services.detection.predictor import model_emits_logits

    # AASIST usa AM-Softmax com saída linear: emite logits crus.
    aasist = create_model_by_name("AASIST", input_shape=(16000, 1), num_classes=1)
    assert model_emits_logits(aasist) is True

    # Sonic Sleuth fecha com ativação: já entrega probabilidades.
    sonic = create_model_by_name("Sonic Sleuth", input_shape=(100, 80, 1),
                                 num_classes=1)
    assert model_emits_logits(sonic) is False

    # Estimadores sem `layers` (sklearn) devolvem None → mantém a heurística.
    assert model_emits_logits(object()) is None


def test_contract_field_overrides_the_value_range_heuristic():
    """O caso de fronteira que a heurística sozinha erraria."""
    from app.domain.services.detection.predictor import normalize_logits_to_probs

    # Logits que POR ACASO parecem probabilidades: dentro de [0,1] e somam ~1.
    ambiguo = np.array([[0.4, 0.6], [0.55, 0.45]])

    # Sem o campo, a heurística conclui "já são probabilidades" e não mexe.
    heuristica = normalize_logits_to_probs(ambiguo)
    np.testing.assert_allclose(heuristica, ambiguo, atol=1e-6)

    # Com o contrato declarando logits, o softmax é aplicado — como o
    # benchmark faz, porque ele olha a arquitetura e não os valores.
    declarado = normalize_logits_to_probs(ambiguo, True)
    assert not np.allclose(declarado, ambiguo)
    np.testing.assert_allclose(declarado.sum(axis=-1), [1.0, 1.0], atol=1e-6)

    # E `False` impede normalização indevida de probabilidades legítimas.
    probs = np.array([[0.2, 0.8]])
    np.testing.assert_allclose(
        normalize_logits_to_probs(probs, False), probs, atol=1e-6
    )
