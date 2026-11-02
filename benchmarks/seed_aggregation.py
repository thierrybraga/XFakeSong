"""Consolida as repetições por semente de uma arquitetura em média ± desvio.

Saiu de ``runner.py`` (P2-15).
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

#: Métricas cujo agregado entre repetições é reportado como média ± desvio.
_SEED_AGGREGATED_METRICS = (
    "accuracy",
    "precision",
    "recall",
    "f1",
    "auc_roc",
    "eer",
    "min_tdcf",
    # `accuracy_at_eer_oracle` é a coluna Acur.@EER das tabelas; sem estar aqui,
    # ela seria a da primeira semente enquanto as vizinhas mostram a média.
    "ece",
    "accuracy_at_eer",
    "accuracy_at_eer_oracle",
    "accuracy_at_calibrated_threshold",
)


def _aggregate_metric_block(blocks: list[Dict[str, Any]]) -> Dict[str, Any]:
    """Média entre repetições + desvio-padrão amostral por métrica."""
    base = dict(blocks[0])
    if len(blocks) == 1:
        return base
    for metric in _SEED_AGGREGATED_METRICS:
        values = [
            float(b[metric])
            for b in blocks
            if isinstance(b.get(metric), (int, float)) and np.isfinite(b[metric])
        ]
        if not values:
            continue
        base[metric] = float(np.mean(values))
        # ddof=1: desvio AMOSTRAL — estamos estimando a variabilidade do
        # procedimento de treino a partir de N execuções, não descrevendo uma
        # população completa.
        base[f"{metric}_seed_std"] = (
            float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        )
        base[f"{metric}_seed_values"] = values
    base["n_seeds"] = len(blocks)
    return base


def _aggregate_seed_runs(runs: list[Dict[str, Any]]) -> Dict[str, Any]:
    """Consolida N repetições da MESMA arquitetura em um resultado único.

    Rigor acadêmico: com uma execução por modelo, o benchmark não distingue
    "arquitetura A é melhor que B" de "esta execução de A foi melhor". As
    métricas viram média ± desvio entre sementes de treino, e cada execução
    fica preservada em `seed_runs` para auditoria.

    O artefato PROMOVIDO é o da PRIMEIRA semente — nunca o da melhor. Escolher
    a melhor execução pelo teste seria seleção no conjunto de teste.
    """
    if len(runs) == 1:
        return runs[0]

    ok_runs = [r for r in runs if r.get("status") == "ok"]
    if not ok_runs:
        base = dict(runs[0])
        base["seed_runs"] = runs
        base["n_seeds"] = len(runs)
        return base

    # A primeira execução BEM-SUCEDIDA é a representativa (artefato promovido).
    base = dict(ok_runs[0])
    base["clean"] = _aggregate_metric_block([r["clean"] for r in ok_runs])

    robustness_keys = sorted({k for r in ok_runs for k in (r.get("robustness") or {})})
    if robustness_keys:
        aggregated_rob: Dict[str, Any] = {}
        for key in robustness_keys:
            blocks = [
                r["robustness"][key]
                for r in ok_runs
                if isinstance((r.get("robustness") or {}).get(key), dict)
            ]
            if blocks:
                aggregated_rob[key] = _aggregate_metric_block(blocks)
        base["robustness"] = aggregated_rob

    base["n_seeds"] = len(ok_runs)
    base["training_seeds"] = [r.get("training_seed") for r in ok_runs]
    base["promoted_artifact_seed"] = ok_runs[0].get("training_seed")
    # `clean`/`robustness` são a MÉDIA entre repetições, mas `scores_clean`,
    # `scores_robustness`, `history` e `efficiency` vêm de `ok_runs[0]`. Ou
    # seja: as tabelas mostram a média e as figuras (ROC, DET, matriz de
    # confusão) e os CSVs de predição mostram UMA execução. Sem este campo,
    # nada no artefato dizia qual — e as legendas não podiam declarar.
    base["scores_seed"] = ok_runs[0].get("training_seed")
    base["seed_runs"] = [
        {
            "training_seed": r.get("training_seed"),
            "status": r.get("status"),
            "clean": r.get("clean"),
            "robustness": r.get("robustness"),
            # O runner grava `wall_time_s` (não `duration_sec`): a chave errada
            # deixava o tempo de cada repetição sempre nulo em `seed_runs`.
            "wall_time_s": r.get("wall_time_s"),
            # MOTIVO DA FALHA (2026-08-18). `_benchmark_one` ja devolve
            # `error` quando a repeticao levanta, mas esta projecao copiava
            # cinco campos e deixava esse de fora -- entao o artefato
            # registrava `status: "error"` sem uma palavra sobre o porque.
            # Observado no estudo de sementes do Conformer: a semente 44
            # falhou em 50,8 s, `n_seeds` caiu de 3 para 2, e nao havia como
            # saber depois o que aconteceu. `None` nas repeticoes que deram
            # certo mantem o schema estavel.
            "error": r.get("error"),
        }
        for r in runs
    ]
    return base
