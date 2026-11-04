"""Medida da fusão entre modelos, usando só a validação.

Cada modelo entra pela pontuação centrada no próprio limiar (`operating_point.fake_score`)
e a fusão é a média simples: 0,5 continua sendo o ponto de decisão e não há peso para ajustar.

Limiares e conjunto saem de dez locutores, e a medida vem do décimo primeiro. Buscar entre
todos os conjuntos, mesmo assim, sobreajusta: medido em 2026-11-02, o conjunto escolhido
assim deu EER de 0,69% em locutor não visto, contra 0,14% da regra simples dos três melhores
modelos. Por isso a escolha é pela regra `top_k` ou por um conjunto fixado antes.

A validação já elegeu a época de cada modelo, então estes números são otimistas. A estimativa
final de um conjunto escolhido é do teste, uma única vez.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import roc_curve

from app.domain.services.detection.operating_point import PONTO_DE_DECISAO

Scores = Dict[str, np.ndarray]
Chooser = Callable[[np.ndarray], Tuple[str, ...]]


def eer_and_threshold(y_true: np.ndarray, scores: np.ndarray) -> Tuple[float, float]:
    """(EER, limiar) no ponto em que FPR e FNR mais se aproximam."""
    fpr, tpr, limiares = roc_curve(y_true, scores)
    fnr = 1.0 - tpr
    i = int(np.argmin(np.abs(fpr - fnr)))
    return float((fpr[i] + fnr[i]) / 2.0), float(limiares[i])


def centered_scores(p_fake: np.ndarray, threshold: float) -> np.ndarray:
    """`operating_point.fake_score` para um vetor de P(fake)."""
    p = np.clip(np.asarray(p_fake, dtype=np.float64), 0.0, 1.0)
    if not 0.0 < threshold < 1.0:
        return p
    return np.where(
        p < threshold,
        PONTO_DE_DECISAO * p / threshold,
        PONTO_DE_DECISAO + PONTO_DE_DECISAO * (p - threshold) / (1.0 - threshold),
    )


def _fixed_chooser(models: Sequence[str]) -> Chooser:
    conjunto = tuple(sorted(models))
    return lambda _ajuste: conjunto


def top_k_chooser(
    calibration: Scores, y_true: np.ndarray, k: int, names: Iterable[str]
) -> Chooser:
    """Os `k` modelos de menor EER nos locutores de ajuste."""
    candidatos = sorted(names)

    def escolher(ajuste: np.ndarray) -> Tuple[str, ...]:
        eers = {
            nome: eer_and_threshold(y_true[ajuste], calibration[nome][ajuste])[0]
            for nome in candidatos
        }
        return tuple(sorted(sorted(candidatos, key=lambda n: eers[n])[:k]))

    return escolher


def held_out_scores(
    calibration: Scores,
    target: Scores,
    y_true: np.ndarray,
    speakers: np.ndarray,
    choose: Chooser,
) -> Tuple[np.ndarray, Dict[Tuple[str, ...], int]]:
    """Pontuação fundida de cada áudio, com limiares e conjunto de outros locutores.

    `calibration` é a condição em que os limiares são derivados, a validação limpa, como no
    sistema implantado. `target` é a condição medida. As rodadas podem escolher conjuntos
    diferentes, e as pontuações se somam porque todas têm o limiar em 0,5.
    """
    locutores = np.asarray(speakers).astype(str).ravel()
    fundida = np.zeros(len(y_true))
    escolhas: Dict[Tuple[str, ...], int] = {}
    for locutor in sorted(set(locutores)):
        fora = locutores == locutor
        ajuste = ~fora
        conjunto = choose(ajuste)
        escolhas[conjunto] = escolhas.get(conjunto, 0) + 1
        fundida[fora] = np.mean(
            [
                centered_scores(
                    target[nome][fora],
                    eer_and_threshold(y_true[ajuste], calibration[nome][ajuste])[1],
                )
                for nome in conjunto
            ],
            axis=0,
        )
    return fundida, escolhas


def _measure(y_true: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    return {
        "eer": eer_and_threshold(y_true, scores)[0],
        "accuracy_at_center": float(((scores >= PONTO_DE_DECISAO) == y_true).mean()),
    }


def compare_fusions(
    calibration: Scores,
    y_true: np.ndarray,
    speakers: np.ndarray,
    conditions: Optional[Dict[str, Scores]] = None,
    fixed_sets: Sequence[Sequence[str]] = (),
    top_ks: Sequence[int] = (1, 2, 3, 4),
) -> List[Dict[str, Any]]:
    """Mede modelos e fusões em locutor não visto, em cada condição.

    Nas condições de ruído só entram os modelos medidos nelas: a regra `top_k` e os
    conjuntos fixos que dependam de outro modelo ficam de fora daquela condição.
    """
    y = np.asarray(y_true).astype(int).ravel()
    todas = {"clean": calibration, **(conditions or {})}
    linhas: List[Dict[str, Any]] = []
    for condicao, target in todas.items():
        disponiveis = sorted(target)
        regras: List[Tuple[str, Chooser]] = [
            (f"model:{nome}", _fixed_chooser((nome,))) for nome in disponiveis
        ]
        regras += [
            (
                f"top-{k}",
                top_k_chooser(
                    {n: calibration[n] for n in disponiveis}, y, k, disponiveis
                ),
            )
            for k in top_ks
            if k <= len(disponiveis)
        ]
        regras += [
            (f"set:{'+'.join(sorted(s))}", _fixed_chooser(s))
            for s in fixed_sets
            if set(s) <= set(disponiveis)
        ]
        for rotulo, escolher in regras:
            fundida, escolhas = held_out_scores(
                calibration, target, y, speakers, escolher
            )
            linhas.append(
                {
                    "condition": condicao,
                    "rule": rotulo,
                    **_measure(y, fundida),
                    "chosen": [
                        {"models": list(c), "rounds": n} for c, n in escolhas.items()
                    ],
                }
            )
    return linhas
