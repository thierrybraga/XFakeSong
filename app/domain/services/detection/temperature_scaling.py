"""Temperature scaling sobre os logits do modelo, com busca contínua.

Dois defeitos motivaram este módulo:

- A busca varria uma grade de 0,5 a 4,9. Conformer parou no piso (T=0,5) e RawGAT-ST no
  teto (T=4,9): a temperatura escolhida era só o limite da grade.
- O logit era recuperado da probabilidade, cortada em 1e-7. Todo áudio com log-odds acima
  de 16,1 virava o mesmo valor, e daí nasce o teto de 0,97891 do AASIST (T=4,2).

A perda é convexa em 1/T, então uma busca limitada em log(T) acha o único mínimo. Medido em
2026-11-02 sobre os 1.456 áudios de validação: o corte atingia 94% das amostras no AST, 86%
no Hybrid CNN-Transformer, 84% no AASIST, 19% no RawGAT-ST e nenhuma no Conformer.

O contrato declara a escala em `temperature_on_logits`. Um artefato sem o campo foi avaliado
na escala cortada e continua nela, reproduzindo os números do seu run. Quando não há acesso
à camada final (ONNX, modelo sem `layers`), o caminho também é o antigo.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit, log_softmax, softmax

logger = logging.getLogger(__name__)

# Limites amplos o bastante para nunca decidirem sozinhos: se a busca encostar neles, o
# modelo está separando a validação quase perfeitamente ou quase nada, e isso vai ao log.
TEMPERATURE_MIN = 0.05
TEMPERATURE_MAX = 100.0

_EPS = 1e-7


def as_class_logits(scores: np.ndarray) -> np.ndarray:
    """Normaliza para (N, K), K >= 2. Uma saída sigmoide vira as classes [0, z]."""
    z = np.asarray(scores, dtype=np.float64)
    if z.ndim == 1:
        z = z[:, np.newaxis]
    if z.shape[-1] == 1:
        z = np.concatenate([np.zeros_like(z), z], axis=-1)
    return z


def logits_from_probabilities(probabilities: np.ndarray) -> np.ndarray:
    """Logits aproximados, para quem só tem a probabilidade. Satura em log-odds 16,1."""
    p = np.clip(np.asarray(probabilities, dtype=np.float64), _EPS, 1.0 - _EPS)
    if p.ndim > 1 and p.shape[-1] > 1:
        return np.log(p)
    return np.log(p / (1.0 - p))


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Temperatura que minimiza a NLL (Guo et al., ICML 2017)."""
    z = as_class_logits(logits)
    y = np.asarray(labels).astype(int).ravel()
    linhas = np.arange(len(y))

    def nll(log_t: float) -> float:
        return float(-log_softmax(z / np.exp(log_t), axis=-1)[linhas, y].mean())

    limites = (np.log(TEMPERATURE_MIN), np.log(TEMPERATURE_MAX))
    resultado = minimize_scalar(
        nll, bounds=limites, method="bounded", options={"xatol": 1e-4}
    )
    temperature = float(np.exp(resultado.x))
    if min(temperature / TEMPERATURE_MIN, TEMPERATURE_MAX / temperature) < 1.01:
        logger.warning(
            "Temperatura %.3f encostou no limite da busca [%s, %s]",
            temperature,
            TEMPERATURE_MIN,
            TEMPERATURE_MAX,
        )
    return temperature


def transported_threshold(
    threshold: float, old_temperature: float, new_temperature: float
) -> float:
    """Limiar equivalente sob outra temperatura: o mesmo log-odds, e o mesmo veredito."""
    limiar = float(np.clip(threshold, _EPS, 1.0 - _EPS))
    log_odds = np.log(limiar / (1.0 - limiar))
    return float(expit(log_odds * old_temperature / new_temperature))


def calibrated_probabilities(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Probabilidades de `logits / T`, em float64. Mantém a forma da entrada."""
    z = np.asarray(logits, dtype=np.float64)
    t = float(temperature) if temperature and temperature > 0 else 1.0
    if z.ndim > 1 and z.shape[-1] > 1:
        return softmax(z / t, axis=-1)
    return expit(z / t)


def logit_output_model(model: Any) -> Optional[Any]:
    """Modelo com os mesmos pesos que devolve o logit, ou ``None`` se não houver acesso.

    Usa `model.layers[-1]`, o mesmo critério de `predictor.model_emits_logits`. A camada
    final é copiada sem ativação: a cópia congela os pesos, então construa depois de
    restaurar o checkpoint.
    """
    try:
        import tensorflow as tf

        if not isinstance(model, tf.keras.Model):
            return None
        head = model.layers[-1]
        activation = getattr(head, "activation", None)
        nome = getattr(activation, "__name__", None)
        if activation is None or nome == "linear":
            return model
        if nome not in {"softmax", "sigmoid"}:
            return None
        if isinstance(head, tf.keras.layers.Activation):
            return tf.keras.Model(model.inputs, head.input)
        if not isinstance(head, tf.keras.layers.Dense):
            return None
        sem_ativacao = tf.keras.layers.Dense(
            head.units,
            activation=None,
            use_bias=head.use_bias,
            dtype=head.dtype_policy,
            name=f"{head.name}_logits",
        )
        saida = sem_ativacao(head.input)
        sem_ativacao.set_weights(head.get_weights())
        return tf.keras.Model(model.inputs, saida)
    except Exception as e:  # noqa: BLE001 — sem logit, o chamador usa a probabilidade
        logger.debug("Logits indisponíveis: %s", e)
        return None
