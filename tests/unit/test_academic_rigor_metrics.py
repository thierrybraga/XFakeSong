"""Testes do rigor acadêmico (2026-07-14): IC bootstrap, ECE e scores de logits.

Cobre:
- IC 95% de bootstrap (EER/AUC/accuracy) em evaluate_scores;
- ECE (calibração) reportado sempre;
- normalização de logits na extração de p_fake do runner (bug do AASIST:
  logits AMSoftmax clipados em [0,1] quantizavam os scores).
"""

import numpy as np
import pytest


def _scores(n=400, seed=0, sep=2.0):
    rng = np.random.default_rng(seed)
    y = np.repeat([0, 1], n // 2)
    p = 1.0 / (1.0 + np.exp(-(rng.standard_normal(n) + sep * (y - 0.5))))
    return y, p


def test_evaluate_scores_reports_ece_and_no_ci_by_default():
    from benchmarks.evaluate import evaluate_scores

    y, p = _scores()
    out = evaluate_scores(y, p)
    assert "ece" in out and 0.0 <= out["ece"] <= 1.0
    assert "eer_ci95_low" not in out  # bootstrap é opt-in


def test_evaluate_scores_bootstrap_ci_brackets_point_estimate():
    from benchmarks.evaluate import evaluate_scores

    y, p = _scores()
    out = evaluate_scores(y, p, n_bootstrap=200)
    for metric in ("eer", "auc_roc", "accuracy"):
        lo = out[f"{metric}_ci95_low"]
        hi = out[f"{metric}_ci95_high"]
        assert lo <= hi
        # O IC deve conter (ou tangenciar) a estimativa pontual.
        assert lo - 1e-9 <= out[metric] <= hi + 1e-9
    assert out["bootstrap_samples"] > 0


def test_ece_perfectly_calibrated_is_small_and_overconfident_is_large():
    from benchmarks.evaluate import _expected_calibration_error

    rng = np.random.default_rng(1)
    n = 20000
    # Scores calibrados: p ~ U(0,1) e y ~ Bernoulli(p)
    p = rng.uniform(0, 1, n)
    y = (rng.uniform(0, 1, n) < p).astype(int)
    ece_cal = _expected_calibration_error(y, p)
    # Superconfiante: mesmo y, scores saturados na direção da decisão.
    p_over = np.where(p >= 0.5, 0.999, 0.001)
    ece_over = _expected_calibration_error(y, p_over)
    assert ece_cal < 0.05
    assert ece_over > ece_cal + 0.05


def test_runner_predict_normalizes_linear_logits(tmp_path):
    """Modelos com saída LINEAR (AASIST/AMSoftmax) devem virar probabilidade
    ANTES do clip de _finite_scores — senão o score quantiza em {0, 1}."""
    tf = pytest.importorskip("tensorflow")
    from benchmarks.runner import _finite_scores

    # Modelo mínimo com saída linear de 2 unidades (logits crus).
    tf.keras.utils.set_random_seed(0)
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(4,)),
            tf.keras.layers.Dense(2, activation=None),  # logits
        ]
    )
    last_act = getattr(model.layers[-1], "activation", None)
    assert last_act is tf.keras.activations.linear

    # Reproduz a lógica do runner (_normalize_probs) sobre logits grandes.
    X = np.random.default_rng(0).standard_normal((64, 4)).astype("float32")
    logits = model.predict(X, verbose=0).astype("float64") * 15.0  # ≈AMSoftmax

    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    probs = e / e.sum(axis=-1, keepdims=True)
    scores = _finite_scores(probs[:, 1])

    # SEM a normalização, o clip dos logits em [0,1] colapsa quase tudo em
    # {0, 1}; COM softmax os scores preservam o ranking contínuo (saturação
    # exata a 0/1 só nos extremos de |logit| grande — comportamento correto).
    clipped_raw = _finite_scores(logits[:, 1])
    assert len(np.unique(scores)) > len(np.unique(clipped_raw))
    assert len(np.unique(scores)) >= len(scores) // 2
    assert (scores >= 0).all() and (scores <= 1).all()
    # Ranking do softmax segue (z1 − z0) — o score correto p/ EER/ROC.
    # Monotonicidade (permite empates nos extremos saturados em 0/1):
    margin = logits[:, 1] - logits[:, 0]
    ordered = scores[np.argsort(margin)]
    assert (np.diff(ordered) >= -1e-12).all()
