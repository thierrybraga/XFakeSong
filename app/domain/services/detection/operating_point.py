"""Pontuação centrada no limiar: 50% é sempre o ponto de decisão do modelo.

Cada modelo decide em `p_fake >= limiar`, e o limiar calibrado vai de 0,05 (SVM) a 0,99
(Hybrid CNN-Transformer). Por isso o mesmo número tinha significados opostos: P(fake) de
0,80 é "real" no AASIST (corte em 0,94) e "fake" no Conformer (corte em 0,48). E a
confiança na classe decidida, `1 - p_fake`, saía 0,20 num veredito "real".

A pontuação reescala P(fake) em dois segmentos lineares: 0 fica em 0, o limiar vai a
0,5 e 1 fica em 1. A ordem dos áudios e todo veredito se mantêm (AUC, EER e decisões não
mudam), e com limiar 0,5 é a identidade. `p_fake` continua disponível para quem precisa
do valor do modelo: este módulo só define como comunicar o resultado.

Por que linear, e não uma translação em logit: medido em 2026-11-02 sobre 200 áudios de
validação em cada um dos 9 modelos. Nos 37 casos em que P(fake) e o veredito apontam para
lados opostos, o acerto real foi de 59,5%; a linear declara 54,6%, e a logit, 66,4%. Nas
demais faixas o acerto real fica acima do declarado, salvo a mais alta (-0,7 ponto).
Mesmo assim, a pontuação não é uma probabilidade calibrada, e não deve ser chamada assim.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

PONTO_DE_DECISAO = 0.5


def fake_score(p_fake: float, threshold: Optional[float] = None) -> float:
    """P(fake) reescalada para que o limiar do modelo fique em 0,5."""
    p = min(max(float(p_fake), 0.0), 1.0)
    limiar = PONTO_DE_DECISAO if threshold is None else float(threshold)
    if not 0.0 < limiar < 1.0:
        # Sem segmentos para reescalar: não há como reposicionar um limiar em 0 ou 1.
        return p
    if p < limiar:
        return PONTO_DE_DECISAO * p / limiar
    return PONTO_DE_DECISAO + PONTO_DE_DECISAO * (p - limiar) / (1.0 - limiar)


def verdict_confidence(p_fake: float, threshold: Optional[float] = None) -> float:
    """Confiança no veredito, entre 0,5 (em cima do limiar) e 1."""
    pontuacao = fake_score(p_fake, threshold)
    return max(pontuacao, 1.0 - pontuacao)


def fuse_scores(
    scores: Sequence[float],
    fusion: str = "soft_voting",
    weights: Optional[Sequence[float]] = None,
) -> float:
    """Combina as pontuações centradas de vários modelos, mantendo o limiar em 0,5.

    Como todas têm o limiar em 0,5, a média preserva o significado do ponto de decisão.
    Em `majority_vote`, a pontuação é a média do lado vencedor, para nunca contradizer o
    voto; a antiga troca pela média perto de 0,5 podia inverter o resultado.
    """
    valores = np.asarray(scores, dtype=np.float64)
    if fusion == "weighted_avg":
        pesos = (
            np.full(len(valores), 1.0 / len(valores))
            if weights is None
            else np.asarray(weights, dtype=np.float64)
        )
        return float(np.sum(valores * pesos) / np.sum(pesos))
    if fusion == "soft_voting":
        return float(np.mean(valores))
    if fusion == "max_conf":
        return float(valores[int(np.argmax(np.abs(valores - PONTO_DE_DECISAO)))])
    if fusion == "majority_vote":
        votos_fake = valores >= PONTO_DE_DECISAO
        if 2 * int(votos_fake.sum()) == len(valores):
            return float(np.mean(valores))
        lado_fake = 2 * int(votos_fake.sum()) > len(valores)
        return float(np.mean(valores[votos_fake == lado_fake]))
    raise ValueError(f"Fusão desconhecida: {fusion}")
